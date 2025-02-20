

import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
from robouniview.models.transformers.ops.uvformer.modules import MSDeformAttn
if 'V100' in torch.cuda.get_device_properties(0).name:
    from robouniview.models.transformers.ops.uvformer_v100.modules import MSDeformAttn
from robouniview.models.transformers.transformer_utils import _get_activation_fn
from robouniview.models.transformers.transformer_utils import _get_clones
from robouniview.models.transformers.transformer_utils import encode_grid_to_emb2d
from robouniview.models.transformers.transformer_utils import create_uv_grid
from robouniview.models.transformers.transformer_utils import project_grid_image
from robouniview.models.transformers.transformer_utils import unpack_calib
from robouniview.models.transformers.transformer_utils import euler_to_Rot
from robouniview.models.transformers.cosformer import CosformerAttention


class DeformableTransformerDecoderLayer(nn.Module):
    """
    implements deformable transformer decoder layer
    """
    def __init__(self, global_config, transformer_config, d_model=256, d_ffn=1024, dropout=0.1, activation='relu',
                 n_levels=4, n_heads=8, n_points=4, self_attn_type='MHA'):
        super().__init__()
        self.self_attn_type = self_attn_type
        self.use_ffn = transformer_config.get('use_ffn', True)

        # cross attention
        self.cross_attn = MSDeformAttn(global_config, d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        if self_attn_type == 'MHA':
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        elif self_attn_type == 'cosformer':
            self.self_attn = CosformerAttention(d_model, n_heads, dropout_rate=dropout)
        elif self_attn_type == 'none':
            self.self_attn = None
        else:
            raise ValueError('Unrecognized self attention type: {}'.format(self_attn_type))
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        if self.self_attn_type == 'MHA':
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        elif self.self_attn_type == 'cosformer':
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)).transpose(0, 1)
        elif self.self_attn_type == 'none':
            tgt2 = None
        else:
            raise ValueError('Unrecognized self attention type: {}'.format(self.self_attn_type))

        if tgt2 is not None:
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        # reference_points, bs x num_query x num_feature x 2, range from 0 to 1
        # tgt bs x num_query x dim
        # For our case, if one grid in uv is projected out of image,
        # its reference point just need to be out of [0,1] so that grid_sample will return 0
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points,
                               src, src_spatial_shapes, level_start_index)
        # TODO devide |Vhit| not hard

        tgt = tgt + self.dropout1(tgt2)
        if self.use_ffn:
            tgt = self.norm1(tgt)
            tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, query_pos=None):
        output = tgt

        for _, layer in enumerate(self.layers):
            output = layer(output, query_pos, reference_points, src, src_spatial_shapes, src_level_start_index)

        return output


class DeformableTransformer(nn.Module):
    """
    implements deformable transformer
    """
    def __init__(self, global_config, transformer_config, freeze_module: bool = False,
                 dynamic_query_embed: bool = False, keep_original_query_embed: bool = False):
        super().__init__()
        self.freeze_module = freeze_module
        self.global_config = global_config
        self.transformer_config = transformer_config
        self.is_onnx = False
        self.use_frustum_refpoints = getattr(global_config, 'use_frustum_refpoints', False)
        self.dynamic_query_embed = dynamic_query_embed

        d_model = transformer_config['hidden_dim']
        self.d_model = d_model
        nhead = transformer_config['nhead']
        num_decoder_layers = transformer_config['num_decoder_layers']
        dim_feedforward = transformer_config['dim_feedforward']
        dropout = transformer_config['dropout']
        activation = 'relu'
        dec_n_points = 4
        task_name = transformer_config['task_name']
        self_attn_type = transformer_config['self_attn_type']

        # init cameras
        # this is hacked for back-compatibility
        if self.global_config.UVformer != {}:
            self.cams = self.global_config.cam_trained.copy()
        else:
            self.cams = self.global_config.Tasks[task_name]['cam_used'] + self.global_config.virtual_cams
        if 'cam_trained' in self.transformer_config:
            self.cams = self.transformer_config['cam_trained']
        self.num_cams = len(self.cams)
        self.num_input_feat = len(self.global_config.image_feature_out)
        #self.orig_to_actual_img_ratio = self.global_config.orig_to_actual_img_ratio

        #preprocess_config = self.global_config.Tasks[task_name]['PreProcessing']

        preprocess_config = transformer_config['PreProcessing']


        self.front, self.rear, self.left, self.right = [preprocess_config[x] for x in
                                                        ['front', 'back', 'left', 'right']]
        ref_z_range = transformer_config['ref_z_range']

        if 'rotate' in transformer_config:
            if transformer_config['rotate']:
                self.uv_range = [-self.front, self.rear, -self.left, self.right, ref_z_range[0], ref_z_range[1]]
            else:
                self.uv_range = [self.front, -self.rear, self.left, -self.right, ref_z_range[1], ref_z_range[0]]
        else:
            self.uv_range = [self.front, -self.rear, self.left, -self.right, ref_z_range[1], ref_z_range[0]]

        self.norm = transformer_config.get('norm', 'gn')
        self.grid_resolution = transformer_config['grid_resolution']
        self.uv_grid = create_uv_grid(self.uv_range, self.grid_resolution, self.use_frustum_refpoints)
        self.uv_x, self.uv_y, self.uv_z = self.uv_grid.shape[:3]
        self.num_queries = self.uv_x * self.uv_y
        self.reference_points_dict = {}
        self.reference_points_valid_dict = {}
        self.valid_weight_dict = {}
        self.setup_input_proj()

        num_feature_levels = self.num_cams * self.num_input_feat * self.uv_z
        decoder_layer = DeformableTransformerDecoderLayer(global_config, transformer_config, d_model=d_model,
                                                          d_ffn=dim_feedforward,
                                                          dropout=dropout,
                                                          activation=activation,
                                                          n_levels=num_feature_levels, n_heads=nhead,
                                                          n_points=dec_n_points,
                                                          self_attn_type=self_attn_type)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers)

        # init cam id position encoding
        self.cam_level_embed = nn.Parameter(torch.Tensor(self.num_cams, d_model))
        self.feat_level_embed = nn.Parameter(torch.Tensor(self.num_input_feat, d_model))
        if not dynamic_query_embed:
            # d_model * 2, will be split later with torch.split()
            self.query_embed = nn.Embedding(self.num_queries, d_model * 2)
        else:
            # positional encoding only
            self.query_embed = nn.Embedding(self.num_queries, d_model)
            if keep_original_query_embed:
                # this query embed needs to be learned
                # this is required if the query embed from LSS is not sufficient
                # for instance, the query embed from LSS obtained from cam2 only
                self.orig_query_embed = nn.Embedding(self.num_queries, d_model)

        # upsample uv feat
        upsample_ratio = transformer_config.get('upsample', 1)
        assert math.log(upsample_ratio, 2).is_integer()
        upsample_layers = []
        while upsample_ratio > 1:
            upsample_layers.append(nn.ConvTranspose2d(d_model, d_model, 4, 2, 1, bias=False))
            upsample_layers.append(nn.BatchNorm2d(d_model))
            upsample_layers.append(nn.ReLU(True))
            upsample_ratio /= 2
        self.upsample = nn.Sequential(*upsample_layers)
        self._reset_parameters()

    def setup_input_proj(self):
        # init input projection
        input_proj_dict = {}
        for cam_lvl in range(self.num_cams):
            # The key of nn.ModuleDict must be string
            if self.norm == 'None':
                input_proj_dict[str(cam_lvl)] = nn.Sequential()
            else:
                input_proj_dict[str(cam_lvl)] = nn.Sequential(
                    nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
                    nn.BatchNorm2d(self.d_model) if self.norm == 'bn' else nn.GroupNorm(32, self.d_model),
                )
        self.input_proj = nn.ModuleDict(input_proj_dict)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def get_input_feature(self, level, x):
        feats = torch.stack([c[level] for c in x], axis=1)  # bs,5cam,c,h,w
        return feats

    def get_input_feature_list(self, level, x):
        feats = [c[level] for c in x]  # num_cam*[bs,c,h,w]
        return feats

    def preprocess_input(self, input_feat):
        src_flatten = []
        pos_flatten = []
        spatial_shapes = []
        mask_flatten = []
        for f in range(self.num_input_feat):
            x = input_feat[f]
            num_cams = len(x)  # num_feat*num_cams*[B, C, H, W]
            for cam_lvl in range(num_cams):
                src = x[cam_lvl]
                bs, _, h, w = src.shape  # B, C, H, W  (B, 256, 12, 20)
                src = self.input_proj[str(cam_lvl)](src)  # B, C, H, W
                pos_embed = encode_grid_to_emb2d(src.shape, src.device)  # B, C, H, W
                spatial_shape = (h, w)
              
                spatial_shapes.append(spatial_shape)

                mask = torch.zeros_like(src[:, 0, :, :]).bool()
                mask = mask.flatten(1)
               
                mask_flatten.append(mask)

                src = src.flatten(2).transpose(1, 2)  # B, C, H, W --> B, C, H*W --> B, H*W, C
                pos_embed = pos_embed.flatten(2).transpose(1, 2)  # B, C, H, W --> B, C, H*W --> B, H*W, C
                lvl_pos_embed = pos_embed + self.cam_level_embed[cam_lvl].view(1, 1, -1) \
                                + self.feat_level_embed[f].view(1, 1, -1)  # B, H*W, C

               
                src_flatten.append(src)
                pos_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, 1)  # B, N*H*W, C
        pos_flatten = torch.cat(pos_flatten, 1)  # B, N*H*W, C
        # mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        return src_flatten, pos_flatten, spatial_shapes, level_start_index

    def generate_reference_points(self, calib, ypr_jitter):
        reference_points = []
        reference_points_valid = []
        for h in range(self.uv_z): # z轴，self.uv_z=5
            multifeat_points = []
            multifeat_points_valid = []
            for f in range(self.num_input_feat):
                cam_points = []
                cam_points_valid = []
                for cam_id in self.cams:
                    cam = cam_id
                    intrinsic = calib[cam]['intrinsic_matrix'].detach().cpu().numpy()
                    extrinsic = calib[cam]['extrinsic_matrix'].detach().cpu().numpy()
                    distortion = calib[cam]['distCoeffs_matrix'].detach().cpu().numpy()
                    if 'extrinsic_ypr' in calib[cam]:
                        extrinsic_ypr = calib[cam]['extrinsic_ypr'].detach().cpu().numpy()
                        new_yaw = extrinsic_ypr[0] + ypr_jitter[0]
                        new_pitch = extrinsic_ypr[1] + ypr_jitter[1]
                        new_roll = extrinsic_ypr[2] + ypr_jitter[2]
                        rotation_matrix = euler_to_Rot(np.deg2rad(new_yaw), np.deg2rad(new_pitch),
                                                       np.deg2rad(new_roll)).T
                        extrinsic[0] = rotation_matrix[0].tolist() + [extrinsic[0][-1]]
                        extrinsic[1] = rotation_matrix[1].tolist() + [extrinsic[1][-1]]
                        extrinsic[2] = rotation_matrix[2].tolist() + [extrinsic[2][-1]]
                    
                    # img_h, img_w, _ = self.global_config.original_image_shapes[cam_id]
                    _, img_h, img_w = calib[cam]['fov'].detach().cpu().numpy()
                    # fov = calib[cam]['fov'].detach().cpu().numpy()
                    # foc = img_h / (2 * np.tan(np.deg2rad(fov) / 2))
                    # intrinsic_edited = np.array([[foc , 0.0, img_h/2], [0.0, foc , img_w/2], [0.0, 0.0, 1.0]]) # 这里需要根据图片大小和fov重新计算内参矩阵
                    dist_cam_pt, valid = project_grid_image(self.uv_grid, intrinsic, extrinsic, distortion, img_h, img_w) # dist_cam_pt时UV点，valid是是否映射到了图片上
                    
                    dist_cam_pt = dist_cam_pt.astype(float) 
                    dist_cam_pt[..., 0] = dist_cam_pt[..., 0] / img_w 
                    dist_cam_pt[..., 1] = dist_cam_pt[..., 1] / img_h
                    dist_cam_pt_h = dist_cam_pt[:, :, h, :]
                    valid = valid[:, :, :, None]
                    valid_h = valid[:, :, h, :]
                    
                    cam_points.append(dist_cam_pt_h[:, :, None])
                    cam_points_valid.append(valid_h[:, :, None])

                cam_points = np.concatenate(cam_points, axis=2)
                cam_points_valid = np.concatenate(cam_points_valid, axis=2)
                cam_points_valid = cam_points_valid.repeat(2, axis=3)
                multifeat_points.append(cam_points)
                multifeat_points_valid.append(cam_points_valid)

            multifeat_points = np.concatenate(multifeat_points, axis=2)  # x, y, num_cam*z, uv
            multifeat_points_valid = np.concatenate(multifeat_points_valid, axis=2)  # x, y, num_cam*z, uv
            reference_points.append(multifeat_points)
            reference_points_valid.append(multifeat_points_valid)
        reference_points = np.concatenate(reference_points, axis=2)
        
        reference_points = reference_points.transpose((2, 3, 0, 1)).reshape(
            (self.num_cams * self.num_input_feat * self.uv_z,
             2, self.num_queries)).transpose((2, 0, 1))
        reference_points_valid = np.concatenate(reference_points_valid, axis=2)
        reference_points_valid = reference_points_valid.transpose((2, 3, 0, 1)).reshape(
            (self.num_cams * self.num_input_feat * self.uv_z,
             2, self.num_queries)).transpose((2, 0, 1))

        reference_points = torch.from_numpy(reference_points).float()
        reference_points_valid = torch.from_numpy(reference_points_valid).bool()
        valid_weight = reference_points_valid[:, :, 0].sum(axis=1)
        if valid_weight.min() == 0:
            valid_weight = torch.clamp(valid_weight, min=1)
        valid_weight = valid_weight[:, None]
        return reference_points, reference_points_valid, valid_weight
    

    
    def forward(self, x, calib, lss_query_embed=None, depth_dist=None, forward_orig_query_embed=False):
        input_feat = []
        input_feat_size = 0
        for idx in range(self.num_input_feat): # self.num_input_feat=1
            feat = self.get_input_feature_list(idx, x)  # [[cam1:bs,c,h,w], [cam2:bs,c,h,w], ...]
            if depth_dist is not None:
                # concat depth dist to feature on the channel dim
                new_feat = []
                for f, depth in zip(feat, depth_dist):
                    upsample_ratio = int(f.shape[-1] // depth.shape[-1])
                    depth = F.interpolate(depth, scale_factor=upsample_ratio, mode="bilinear", align_corners=True)
                    new_feat.append(torch.concat([f, depth], 1))
                feat = new_feat
            input_feat.append(feat)
            for f in feat:
                bs, c, feat_h, feat_w = f.shape
                input_feat_size += feat_h * feat_w

        
        src_flatten, pos_flatten, spatial_shapes, level_start_index = self.preprocess_input(input_feat) # input_feat[0][0]: [5*12, 1024,16,16] # 准备deformable输入包括，feat,pos,mask,size

        if not self.dynamic_query_embed:
            query_embeds = self.query_embed.weight
            query_embed, tgt = torch.split(query_embeds, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1) # q的pose和embeding,400*1024
        else:
            # positional encoding
            query_embed = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
            # query
            _, in_channel, _, _ = lss_query_embed.shape
            tgt = lss_query_embed.view(bs, in_channel, -1).permute(0, 2, 1) # BCHW -> B(HW)C
            if forward_orig_query_embed:
                orig_query_embed = self.orig_query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
                tgt += orig_query_embed

     
        unpacked_calibs = unpack_calib(calib, bs)
        reference_points, reference_points_valid, valid_weight = [], [], []
        for i, curr_calib in enumerate(unpacked_calibs): # for bs
            extrinsic_jitter = self.global_config.extrinsic_jitter
            if extrinsic_jitter > 0:
                if self.transformer_config['reuse_ref_point']:
                    # if reuse, we use the same ypr and bigger step
                    step = 0.1
                    all_choice = np.arange(start=-extrinsic_jitter, stop=extrinsic_jitter + step, step=step)
                    jitter = round(np.random.choice(all_choice), 1)
                    ypr_jitter = [jitter, jitter, jitter]
                else:
                    step = 0.01
                    all_choice = np.arange(start=-extrinsic_jitter, stop=extrinsic_jitter + step, step=step)
                    yaw_jitter = round(np.random.choice(all_choice), 2)
                    pitch_jitter = round(np.random.choice(all_choice), 2)
                    roll_jitter = round(np.random.choice(all_choice), 2)
                    ypr_jitter = [yaw_jitter, pitch_jitter, roll_jitter]
            else:
                if hasattr(self.global_config, 'eval_jitter'):
                    eval_jitter = self.global_config.eval_jitter
                    ypr_jitter = [eval_jitter] * 3
                else:
                    ypr_jitter = [0.0, 0.0, 0.0]

            if self.transformer_config['reuse_ref_point']:
                vehicle_name_key = curr_calib['vehicle_name'] + '_%.1f' % ypr_jitter[0]
                if vehicle_name_key not in self.reference_points_dict.keys():
                    curr_reference_points, curr_reference_points_valid, curr_valid_weight = \
                        self.generate_reference_points(curr_calib, ypr_jitter)
                    self.reference_points_dict[vehicle_name_key] = curr_reference_points
                    self.reference_points_valid_dict[vehicle_name_key] = curr_reference_points_valid
                    self.valid_weight_dict[vehicle_name_key] = curr_valid_weight
                else:
                    curr_reference_points = self.reference_points_dict[vehicle_name_key]
                    curr_reference_points_valid = self.reference_points_valid_dict[vehicle_name_key]
                    curr_valid_weight = self.valid_weight_dict[vehicle_name_key]
            else:
                curr_reference_points, curr_reference_points_valid, curr_valid_weight = \
                    self.generate_reference_points(curr_calib, ypr_jitter)
                


                # if i ==0:
                #     curr_reference_points2, curr_reference_points_valid2, curr_valid_weight2 = \
                #         self.save_reference_points(curr_calib, ypr_jitter)
            reference_points.append(curr_reference_points)
            reference_points_valid.append(curr_reference_points_valid)
            valid_weight.append(curr_valid_weight)

        reference_points = torch.stack([i.to(src_flatten.device) for i in reference_points], dim=0)
        reference_points_valid = torch.stack([i.to(src_flatten.device) for i in reference_points_valid], dim=0)
        valid_weight = torch.stack([i.to(src_flatten.device) for i in valid_weight], dim=0)
        reference_points[~reference_points_valid] = -99

        n_ref = self.uv_z
        spatial_shapes = spatial_shapes.repeat(n_ref, 1)
      
        n_ref = self.uv_z
        src_flatten = src_flatten.repeat(1, n_ref, 1)
        tmp = []
        for i in range(n_ref):
            next_rp_level_start_index = (level_start_index + i * input_feat_size).clone().detach()
            tmp.append(next_rp_level_start_index)
        level_start_index = torch.cat(tmp)



        hs = self.decoder(tgt, reference_points, src_flatten, spatial_shapes, level_start_index, query_embed)

       
        hs = hs / valid_weight

        uv_feat = hs.permute(0, 2, 1).contiguous()  # b, c, hw
        uv_feat = uv_feat.reshape(bs, c, self.uv_x, self.uv_y)
        # upsample before spatial alignment
        return uv_feat


# if __name__ == '__main__':
#     uv_t = DeformableTransformer().cuda()
#     x = torch.randn((2, 5, 256, 12, 20)).cuda()
#     uv = uv_t(x)
