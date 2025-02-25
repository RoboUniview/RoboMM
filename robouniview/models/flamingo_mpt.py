import torch
from einops import rearrange, repeat
from torch import nn
import copy
from open_flamingo.src.helpers import PerceiverResampler
from robouniview.models.action_head import DeterministicDecoder, DiffusionDecoder, FCDecoder, GPTDecoder, TokenFCDecoder, Multi_Action_Token_FCDecoder
from robouniview.models.transformers.uvformer import DeformableTransformer
from robouniview.models.transformers.position_encoding import PositionEmbeddingSine, RotaryPositionEncoding3D, get_2d_sincos_pos_embed
from collections import namedtuple
import yaml
import argparse
import copy
import yaml
import numpy as np
import cv2, time
import pybullet as p
from robouniview.models.transformers.petr import PETR, inverse_sigmoid
from robouniview.models.loss_func import (
    FocalLoss, Balanced_BCE_loss, CELoss, BinaryDiceLoss, CELossIgnoreSem, l1_loss)
from robouniview.models.vision_transformer import Block
from robouniview.models.occ_head import FastEncoderHead, Upsample2d_3d, Upsample2d_3d_UVFormer, Decoder_3d, Decoder_3d_4s, Upsample2d_3d_tiny, Decoder_3d_tiny
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

class MPTFlamingo(nn.Module):
    def __init__(
        self,
        args,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        use_media_placement_augmentation: bool = False,
        # this is the window size sampled from the episode
        window_size: int = 8,
        use_gripper=False,
        fusion_mode='',
        sep_resampler=False,
        use_state=False,
        use_diff=False,
        diff_horizon=32,
        last_action=False,
        n_timesteps=150,
        state_dim=15,
        use_hist=False,
        debug=False,
        predict_epsilon=True,
        pad_length=-1,
        multi_step_action=1,
        sep_lm_head=False,
        return_feature = False,
        llm='llama',
        pooling='max',
        residual=False,
        tcp_rel=False,
        replan=-1,
        decoder_type='lstm',
        hidden_size=None,
        fwd_pred=False,
        fwd_pred_hand=False,
        global_latent=10,
        no_image_patch=False,
        refresh=-1,
        nclass_gripper=1,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            use_media_placement_augmentation (bool, optional): Whether to randomly assign images to the preceding or following text in training. Defaults to False.
        """
        super().__init__()
        self.args = args
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size
        self.occ_loss =  args.occ_loss
        self.occ_loss_weight = self.args.occ_loss_weight
        self.train_action = args.train_action
        self.fusion_mode = fusion_mode
        self.vis_dim = vis_dim

        # decode image
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.lang_dim))
        self.decoder_pos_embed_static = nn.Parameter(torch.zeros(1, (112//14)**2, self.lang_dim), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed_static = get_2d_sincos_pos_embed(self.decoder_pos_embed_static.shape[-1], (112//14))
        self.decoder_pos_embed_static.data.copy_(torch.from_numpy(decoder_pos_embed_static).float().unsqueeze(0))
        self.decoder_pos_embed_gripper = nn.Parameter(torch.zeros(1, (112//14)**2,self.lang_dim), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed_gripper = get_2d_sincos_pos_embed(self.decoder_pos_embed_gripper.shape[-1], (112//14))
        self.decoder_pos_embed_gripper.data.copy_(torch.from_numpy(decoder_pos_embed_gripper).float().unsqueeze(0))
        self.decoder_embed = nn.Linear(self.lang_dim, self.lang_dim, bias=True)
        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([
            Block(self.lang_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(self.lang_dim)
        self.decoder_pred = nn.Linear(self.lang_dim, 14**2 * 3, bias=True) # decoder to patch
        
        # encoder 
        if "UVFormer" in self.fusion_mode:
            z_range = self.args.config['UVformer']['transformer_config']['ref_z_range']
            assert int((z_range[1]-z_range[0]) / self.args.config['UVformer']['transformer_config']['grid_resolution'][-1]) == 5
            self.Upsample2d_3d_UVFormer = Upsample2d_3d_UVFormer(in_channels=1024, out_channels=128, z=10) # UVFormer仅使用了两次上采样，从20*20*10-》80*80*40
            self.uvformer = DeformableTransformer(self.args, self.args.UVformer['transformer_config'])
            self.occ_decoder_UVFormer = Decoder_3d_4s()
            if self.occ_loss:
                self.balanced_bce_loss = Balanced_BCE_loss(1, reduction="mean",)
            if hasattr(self.args, 'alignment_layer') and self.args.alignment_layer == 'Linear':
                self.alignment_layer = nn.Linear(self.vis_dim, self.vis_dim)
            elif hasattr(self.args, 'alignment_layer') and self.args.alignment_layer == 'Resampler':
                self.alignment_layer = PerceiverResampler(dim=self.vis_dim)
        else:
            self.rgb = nn.Embedding(1,1024)
            self.gripper = nn.Embedding(1,1024)
            
            # decoder occ
            self.decoder_pos_embed_obs = nn.Parameter(torch.zeros(1, (10)**2,self.lang_dim), requires_grad=False)  # (1, n_patch, h)
            decoder_pos_embed_obs = get_2d_sincos_pos_embed(self.decoder_pos_embed_obs.shape[-1], (10))
            self.decoder_pos_embed_obs.data.copy_(torch.from_numpy(decoder_pos_embed_obs).float().unsqueeze(0))
            self.decoder_embed_obs = nn.Linear(self.lang_dim, self.lang_dim, bias=True)
            self.decoder_blocks_obs = nn.ModuleList([
                Block(self.lang_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
                for i in range(decoder_depth)])
            self.occ_decoder = Decoder_3d()
            z_range = self.args.config['UVformer']['transformer_config']['ref_z_range']
            assert int((z_range[1]-z_range[0]) / self.args.config['UVformer']['transformer_config']['grid_resolution'][-1]) == 5
            self.Upsample2d_3d = Upsample2d_3d(in_channels=2048, out_channels=128, z=5) 
        # encoder 公用
        self.petr = PETR(**self.args.PETR) if hasattr(self.args, 'PETR') else PETR() # 当UVFormer的时候仅有夹爪位置
        # self.split = nn.Unfold(kernel_size=(14, 14), stride=(14, 14), padding=(0, 0))

        if hasattr(self.args, 'T_Embedding') and self.args.T_Embedding :  # 未使用
            self.T_Embedding_layer = nn.Sequential(
                nn.Linear( 1 , self.vis_dim*4),
                nn.ReLU(),
                nn.Linear(self.vis_dim*4, self.vis_dim),
            )


        self.position_embedding = PositionEmbeddingSine(self.args.UVformer["transformer_config"]["hidden_dim"]/2, normalize=True)
        # self.pos_rgb, self.pos_gripper = None, None
        self.use_gripper = use_gripper
        self.use_state = use_state
        
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = use_media_placement_augmentation
        
        self.window_size = window_size
        self.tcp_rel = tcp_rel
        self.act_step = multi_step_action
        print('window size: {}'.format(window_size))
        self.vision_encoder = vision_encoder
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.sep_resampler = sep_resampler
        self.use_hist = use_hist
        self.lang_encoder = lang_encoder
        self.pad_length = pad_length
        self.replan = replan
        if self.replan != -1:
            self.replan = min(int(replan * self.window_size), 180)
        self.refresh = refresh
        

        self.residual = residual
        print(self.vis_dim, self.lang_dim)
        print(lang_encoder.config)
        if not debug:
            if 'llama' in llm:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    use_media_placement_augmentation=self.use_media_placement_augmentation,
                    residual=residual,
                )
            else:
                self.lang_encoder.init_flamingo(
                    media_token_id=media_token_id,
                    lang_hidden_size=self.lang_dim,
                    vis_hidden_size=self.vis_dim,
                    cross_attn_every_n_layers=cross_attn_every_n_layers,
                    gradient_checkpointing=False,
                )
        self.vlm = copy.deepcopy(self.lang_encoder)
        self.vlm_perceiver = copy.deepcopy(self.perceiver)

        if sep_resampler:
            self.perceiver_gripper = PerceiverResampler(dim=self.vis_dim)
            self.perceiver_gripper.load_state_dict(copy.deepcopy(self.perceiver.state_dict()))
        if use_state:
            self.state_fc = nn.Linear(state_dim, self.vis_dim)
        if use_hist:
            self.frame_embs = nn.Parameter(torch.randn(self.window_size, self.vis_dim))
        # To-do: nn archiecture for actor
        self.llm = llm
        if llm=='llama':
            in_features = lang_encoder.lm_head.in_features
        else:
            in_features = self.lang_dim
        self.use_diff = use_diff
        self.decoder_type = decoder_type
        if decoder_type == 'lstm':
            lm_head = DeterministicDecoder(in_features, self.window_size, 
            use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, pooling=pooling)
            self.lang_encoder.lm_head = lm_head
        elif decoder_type == 'fc':
            if use_hist:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            elif 'vit_concat' in fusion_mode:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            elif self.args.multi_action_token :
                self.lang_encoder.lm_head = self.action_head = Multi_Action_Token_FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action, nclass_gripper=nclass_gripper)
            elif self.args.action_token :
                self.lang_encoder.lm_head = self.action_head = TokenFCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
            else:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size, 
                use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, use_state=use_state, return_feature=return_feature, multi_step_action=multi_step_action)
                #raise NotImplementedError
        elif decoder_type == 'diffusion':
            if use_diff:
                self.diffusion_model = DiffusionDecoder(
                    self.action_head.hidden_size, 
                    self.window_size,
                    input_dim=self.action_head.out_features+1,
                    n_timesteps=n_timesteps,
                    horizon=diff_horizon,
                    predict_epsilon=predict_epsilon,
                )
            else:
                raise NotImplementedError
        elif decoder_type=='gpt':
            lm_head = GPTDecoder(in_features, self.window_size, use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode, multi_step_action=multi_step_action, pooling=pooling, hidden_size=hidden_size)
            self.lang_encoder.lm_head = self.action_head = lm_head
        else:
            raise NotImplementedError
        sep_lm_head = True
        self.sep_lm_head = sep_lm_head
        if sep_lm_head:
            self.lm_head = self.lang_encoder.lm_head
            self.lang_encoder.lm_head = nn.Identity()
        self.env = None

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        use_cached_vision_x: bool = False,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
        vision_gripper = None,
        state_tensor = None,
        calib = None,
        pcd = None,
        action_mask = None,
        static_mask = None,
        gripper_mask = None,
        obs_mask = None,
        return_feature = False,
        policy_mask=None
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        st_yf = time.time(); TESTTIME=False

        # raw_rgb = vision_x.clone()
        # raw_gripper = vision_gripper.clone()
        self.pcd = pcd
        assert (vision_x is not None) or use_cached_vision_x, ("Must provide either vision_x or use_cached_vision_x to True.")

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (vision_x is None), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()
        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            if self.use_hist: self._encode_history_vision_post_fusion(vision_x, vision_gripper)
            if "UVFormer" in self.fusion_mode: self._encode_multi_vision_UVformer_fusion(vision_x, vision_gripper, calib, state_tensor = state_tensor)
            else: self._encode_temporal_fusion(vision_x, vision_gripper, calib, state_tensor = state_tensor)

        kwarg = {}
        output, static_pred, gripper_pred, obs_pred, aux_loss = [], None, None, None, {}
        if 0: # I对IL,O对IL，A对AL, L对L
            with torch.no_grad():
                _mask = torch.zeros_like(attention_mask)
                query_mask_matrix = repeat(torch.zeros_like(_mask), 'b n->b m n', m=attention_mask.shape[1])
                if static_mask is not None:
                    static_I_mask = static_mask.clone()
                    static_I_mask[:, :-1] |= static_mask[:, 1:]
                    static_I_mask[:, 1:] |= static_mask[:, :-1]
                    _mask |= static_I_mask
                    query_mask_matrix = query_mask_matrix | torch.einsum('mi,mj->mij', static_I_mask, static_I_mask)
                if gripper_mask is not None:
                    gripper_I_mask = gripper_mask.clone()
                    gripper_I_mask[:, :-1] |= gripper_mask[:, 1:]
                    gripper_I_mask[:, 1:] |= gripper_mask[:, :-1]
                    _mask |= gripper_I_mask
                    query_mask_matrix = query_mask_matrix | torch.einsum('mi,mj->mij', gripper_I_mask, gripper_I_mask)
                if obs_mask is not None:
                    obs_O_mask = obs_mask.clone()
                    obs_O_mask[:, :-1] |= obs_mask[:, 1:]
                    obs_O_mask[:, 1:] |= obs_mask[:, :-1]
                    _mask |= obs_O_mask
                    query_mask_matrix = query_mask_matrix | torch.einsum('mi,mj->mij', obs_O_mask, obs_O_mask)
                query_mask_matrix = query_mask_matrix | ~_mask[:, None] | ~_mask[..., None] # YF: 除了IO外其他都是相关attention的    
                del static_I_mask, gripper_I_mask, obs_O_mask, _mask
                kwarg['query_mask_matrix'] = query_mask_matrix
        if TESTTIME: ed_ecd_yf = time.time(); print(f"YF: image encoder {ed_ecd_yf-st_yf}")
        if self.train_action:
            output = self.vlm(
                input_ids=lang_x,
                attention_mask=attention_mask.bool(),
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True,
                **kwarg,
            )
            output_hs = output.hidden_states[-1] # YF: 3*505*2048
            b, t, d = output_hs.shape
            mask_indices = action_mask.nonzero(as_tuple=True) # 获取掩码为True的元素及其索引
            action_feature = output_hs[mask_indices[0], mask_indices[1]] # 根据掩码从output_hs中选值; 288*20248
            action_feature = rearrange(action_feature, "(B T) D -> B T D", B=b) # 8表示8个token

            vla_lang_x = torch.tensor([self.media_token_id, 50310, 50279]*action_feature.shape[1]+[0], dtype=lang_x.dtype, device=action_feature.device).view(1, -1).repeat(len(action_feature), 1) # 此处方便期间先试用数据。50310表示需要替换为输入特征，50279位read out token
            action_feature = repeat(action_feature, "B T D -> (B N T) D", N = 1)
            kwarg = {"repeat": {"repeat_id":50310, "repeat_feature":action_feature}}
            output = self.lang_encoder(
                input_ids=vla_lang_x,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_hidden_states=True,
                **kwarg,
            )
            output_hs = output.hidden_states[-1] # YF: 3*505*2048
            action_mask = vla_lang_x==50279
            b, t, d = output_hs.shape
            
            if TESTTIME: ed_fd_yf = time.time(); print(f"YF: lang forward {ed_fd_yf-ed_ecd_yf}")
            if static_mask is not None:
                mask_indices = static_mask.nonzero(as_tuple=True) # 获取掩码为True的元素及其索引
                static_feature = output_hs[mask_indices[0], mask_indices[1]] # 根据掩码从output_hs中选值; 288*20248
                static_feature = rearrange(static_feature, "(B T N) D -> B T N D", B=b, N=8) # 8表示8个token
                B, T, N, D = static_feature.shape
                
                mask_tokens = self.mask_token.repeat(B, T, (112//14)**2, 1)  # (b, l, n_patches, h)
                mask_tokens = mask_tokens +  self.decoder_pos_embed_static.unsqueeze(0).repeat(B, T, 1, 1)  # (b, l, n_patches, h)
                static_feature = self.decoder_embed(static_feature) 
                static_pred_ = torch.cat([static_feature, mask_tokens], dim=2)

                static_pred_ = static_pred_.reshape(-1, static_pred_.shape[-2], static_pred_.shape[-1])  # (b * l, n_patches + n_patch_latens, h)
                for blk in self.decoder_blocks:
                    static_pred_ = blk(static_pred_)
                static_pred_ = self.decoder_norm(static_pred_)
                static_pred = self.decoder_pred(static_pred_) 
                static_pred = static_pred.reshape(B, T, -1, static_pred.shape[-1])  # (b, l, n_patches + n_patch_latens, h)
                static_pred = static_pred[:, :, -(112//14)**2:]  # (b, l, n_patches, h)
                static_pred = rearrange(static_pred, "B T N H -> (B T) H N")

                unsplit = nn.Fold(output_size=(112, 112), kernel_size=(14, 14), stride=(14, 14), padding=(0, 0))
                static_pred = unsplit(static_pred)

            if gripper_mask is not None:
                mask_indices = gripper_mask.nonzero(as_tuple=True) # 获取掩码为True的元素及其索引
                gripper_feature = output_hs[mask_indices[0], mask_indices[1]] # 根据掩码从output_hs中选值; 288*20248
                gripper_feature = rearrange(gripper_feature, "(B T N) D -> B T N D", B=b, N=8) # 8表示8个token
                B, T, N, D = gripper_feature.shape

                mask_tokens = self.mask_token.repeat(B, T, (112//14)**2, 1)  # (b, l, n_patches, h)
                mask_tokens = mask_tokens +  self.decoder_pos_embed_gripper.unsqueeze(0).repeat(B, T, 1, 1)  # (b, l, n_patches, h)
                gripper_feature = self.decoder_embed(gripper_feature) 
                gripper_pred_ = torch.cat([gripper_feature, mask_tokens], dim=2)

                gripper_pred_ = gripper_pred_.reshape(-1, gripper_pred_.shape[-2], gripper_pred_.shape[-1])  # (b * l, n_patches + n_patch_latens, h)
                for blk in self.decoder_blocks:
                    gripper_pred_ = blk(gripper_pred_)
                gripper_pred_ = self.decoder_norm(gripper_pred_)
                gripper_pred = self.decoder_pred(gripper_pred_) 

                gripper_pred = gripper_pred.reshape(B, T, -1, gripper_pred.shape[-1])  # (b, l, n_patches + n_patch_latens, h)
                gripper_pred = gripper_pred[:, :, -(112//14)**2:]  # (b, l, n_patches, h)
                gripper_pred = rearrange(gripper_pred, "B T N H -> (B T) H N")
            
                unsplit = nn.Fold(output_size=(112, 112), kernel_size=(14, 14), stride=(14, 14), padding=(0, 0))
                gripper_pred = unsplit(gripper_pred)

            if obs_mask is not None:
                mask_indices = obs_mask.nonzero(as_tuple=True) # 获取掩码为True的元素及其索引
                obs_feature = output_hs[mask_indices[0], mask_indices[1]] # 根据掩码从output_hs中选值; 288*20248
                obs_feature = rearrange(obs_feature, "(B T N) D -> B T N D", B=b, N=8) # 8表示8个token
                B, T, N, D = obs_feature.shape

                mask_tokens = self.mask_token.repeat(B, T, (10)**2, 1)  # (b, l, n_patches, h)
                mask_tokens = mask_tokens +  self.decoder_pos_embed_obs.unsqueeze(0).repeat(B, T, 1, 1)  # (b, l, n_patches, h)
                obs_feature = self.decoder_embed_obs(obs_feature) 
                obs_pred_ = torch.cat([obs_feature, mask_tokens], dim=2) 

                obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])  # (b * l, n_patches + n_patch_latens, h)
                for blk in self.decoder_blocks_obs:
                    obs_pred_ = blk(obs_pred_)
                obs_pred_ = obs_pred_[:, -(10)**2:]  # (b, l, n_patches, h)
                obs_pred_ =  rearrange(obs_pred_, " B (H W) D-> B D H W", H=10, W=10)
                # obs_pred_ =  rearrange(obs_pred_, " B H W D -> B D H W")
                obs_pred_ = self.Upsample2d_3d(obs_pred_) 
                obs_pred = self.occ_decoder(obs_pred_[0])
                obs_pred = rearrange(obs_pred, "BT C Z H W -> BT H W Z C")
            
             # self.decoder_pos_embed_static  
            if TESTTIME: ed_rc_yf = time.time(); print(f"YF: image decoder {ed_rc_yf-ed_fd_yf}")
            if self.args.action_token or self.args.multi_action_token:
                output_hs = self.lm_head(output_hs, state_tensor=state_tensor, action_mask = action_mask)
            else:
                output_hs = self.lm_head(output_hs, state_tensor=state_tensor)
            output.logits = output_hs
            if TESTTIME: ed_yf = time.time(); print(f"YF: loss {ed_yf-ed_rc_yf}")
            if obs_pred is None and self.occ_loss: obs_pred = self.occ
        
        if self.occ_loss and self.pcd is not None:
            aux_loss['loss_occ'] = self.loss_occ()
            
        return output, static_pred, gripper_pred, obs_pred, aux_loss

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert False
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_vision(self, vision_x: torch.Tensor, state_tensor=None):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad() and autocast(): # 这里使用autocast混合精度训练
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

            
    def _encode_temporal_fusion(self, vision_rgb: torch.Tensor, # [b,t,1,1,c,h,w]
                                    vision_gripper: torch.Tensor,  calib, state_tensor=None):
        # if hasattr(self.args, 'T_Embedding') and self.args.T_Embedding :
        #     self.T_Embedding_layer = nn.Sequential(
        #         nn.Linear( 1 , self.vis_dim*4),
        #         nn.ReLU(),
        #         nn.Linear(self.vis_dim*4, self.vis_dim),
        #     )
        TESTTIME=False; st_yf=time.time()
        vision_rgb = vision_rgb.squeeze(2)
        vision_gripper = vision_gripper.squeeze(2)

        with torch.no_grad():
            vision_rgb = self._encode_vision(vision_rgb) #  b T F v d. # 注意这里不算梯度
            vision_gripper = self._encode_vision(vision_gripper)
        # vision_gripper = vision_gripper.clone().detach()
        # vision_rgb = vision_rgb.clone().detach()
        B, T, F, v, d  = vision_rgb.shape

        if hasattr(self.args, 'T_Embedding') and self.args.T_Embedding :
            for t in range(T):
                T_Embedding = self.T_Embedding_layer(inverse_sigmoid(torch.tensor(t*1.0/T)).unsqueeze(0).to(vision_rgb.device))
                vision_rgb[:,t,...] = vision_rgb[:,t,...] + T_Embedding
                vision_gripper[:,t,...] = vision_gripper[:,t,...] + T_Embedding

        vision_gripper = rearrange(vision_gripper, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
        vision_rgb = rearrange(vision_rgb, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)

        if TESTTIME: ed_en_yf=time.time(); print(f"YF: image encoder {ed_en_yf-st_yf}")

        if calib is not None:
            # assert self.window_size==12
            _calib1 = {'rgb_static':{'extrinsic_matrix':rearrange(calib[0][:,:self.window_size]," B T H W -> (B T) H W").cpu(),  # 注意此处最多是window_size个
                                    'intrinsic_matrix':rearrange(calib[1][:,:self.window_size]," B T H W -> (B T) H W").cpu(),
                                    'distCoeffs_matrix':rearrange(calib[2][:,:self.window_size]," B T H -> (B T) H").cpu(),
                                    'fov':rearrange(calib[7][:,:self.window_size]," B T H -> (B T) H").cpu()},
                        'rgb_gripper':{'extrinsic_matrix':rearrange(calib[3][:,:self.window_size]," B T H W -> (B T) H W").cpu(),
                                        'intrinsic_matrix':rearrange(calib[4][:,:self.window_size]," B T H W -> (B T) H W").cpu(),
                                        'distCoeffs_matrix':rearrange(calib[5][:,:self.window_size]," B T H -> (B T) H").cpu(),
                                        'fov':rearrange(calib[8][:,:self.window_size]," B T H -> (B T) H").cpu()}}
            if TESTTIME: ed_petr_yf=time.time(); print(f"YF: petr forward1 {ed_petr_yf-ed_en_yf}")
            with torch.no_grad():
                # if self.pos_rgb is None or self.pos_rgb.shape != vision_rgb.shape:
                pos_rgb = self.position_embedding(vision_rgb)
                # if self.pos_gripper is None or self.pos_gripper.shape != vision_gripper.shape:
                pos_gripper = self.position_embedding(vision_gripper)
                # pos_rgb, pos_gripper = pos_rgb.detach(), pos_gripper.detach()

            pos_embed_rgb = self.petr(vision_rgb, _calib1['rgb_static'], pos_rgb, 200, 200, 0)
            pos_embed_gripper = self.petr(vision_gripper, _calib1['rgb_gripper'], pos_gripper, 84, 84, 0)
            vision_rgb = vision_rgb+pos_embed_rgb
            vision_gripper = vision_gripper+pos_embed_gripper

            vision_rgb = rearrange(vision_rgb, " (B T F) C BH BW ->B T F (BH BW) C", B=B, T=T)
            vision_gripper = rearrange(vision_gripper, " (B T F) C BH BW ->B T F (BH BW) C", B=B, T=T)

        if TESTTIME: ed_petr_yf=time.time(); print(f"YF: petr forward {ed_petr_yf-ed_en_yf}")

        vision_rgb = vision_rgb + self.rgb.weight[0]
        vision_gripper = vision_gripper + self.gripper.weight[0] 
        #forward(self, mlvl_feats, calibs, pos, pad_h, pad_w,fov)

        vision_rgb = self.perceiver(vision_rgb)
        vision_gripper = self.perceiver(vision_gripper)

        if self.args.multi_action_token:
            pass
        else:
            vision_rgb = rearrange(vision_rgb, " B T N D -> B (T N) D")
            vision_gripper = rearrange(vision_gripper, " B T N D -> B (T N) D")
            vision_rgb = vision_rgb.unsqueeze(1)
            vision_gripper = vision_gripper.unsqueeze(1)

        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2) #B T 2N D
       # vision_x = rearrange(vision_x, "B T (I N) D ->B (T I) N D", I = 2)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        feats = {}
        if TESTTIME: ed_yf=time.time(); print(f"YF: image insert {ed_yf-ed_petr_yf}")

        return vision_x, feats
    
    def _encode_multi_vision_UVformer_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, calib, state_tensor=None):
        
        vision_rgb = vision_rgb.squeeze(2)
        vision_gripper = vision_gripper.squeeze(2)
        dtype = vision_rgb.dtype
        with torch.no_grad():
            vision_rgb = self._encode_vision(vision_rgb).to(dtype)
            vision_gripper = self._encode_vision(vision_gripper).to(dtype)
        B, T, F, HxW, C = vision_rgb.shape

        if 1:
            vlm_vision_rgb = self.vlm_perceiver(vision_rgb)  # B T HW C
            vlm_vision_gripper = self.vlm_perceiver(vision_gripper)

            vlm_vision_x = torch.cat([vlm_vision_rgb, vlm_vision_gripper], dim=2)  # reshapes to (b, T, 2*n, d)
            for layer in self.vlm._get_decoder_layers():
                layer.condition_vis_x(vlm_vision_x)

        vision_rgb = rearrange(vision_rgb, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)
        vision_gripper = rearrange(vision_gripper, " B T F (H W) C  -> (B T F) C H W", H=16, W=16)

        _calib1 = {'rgb_static':{'extrinsic_matrix':rearrange(calib[0][:,:self.window_size]," B T H W -> (B T) H W").cpu(),  # 注意此处最多是window_size个
                                'intrinsic_matrix':rearrange(calib[1][:,:self.window_size]," B T H W -> (B T) H W").cpu(),
                                'distCoeffs_matrix':rearrange(calib[2][:,:self.window_size]," B T H -> (B T) H").cpu(),
                                'fov':rearrange(calib[7][:,:self.window_size]," B T H -> (B T) H").cpu()},
                    'rgb_gripper':{'extrinsic_matrix':rearrange(calib[3][:,:self.window_size]," B T H W -> (B T) H W").cpu(),
                                    'intrinsic_matrix':rearrange(calib[4][:,:self.window_size]," B T H W -> (B T) H W").cpu(),
                                    'distCoeffs_matrix':rearrange(calib[5][:,:self.window_size]," B T H -> (B T) H").cpu(),
                                    'fov':rearrange(calib[8][:,:self.window_size]," B T H -> (B T) H").cpu()}}
        x = [[vision_rgb], [vision_gripper]]
        uv_feat = self.uvformer(x, _calib1)  # 20*20*10
        
        if self.occ_loss:
            occ_feat = uv_feat.clone() 
            occ_feat, _ = self.Upsample2d_3d_UVFormer(occ_feat)#(B*T, C, Z, H, W) 
            self.occ = self.occ_decoder_UVFormer(occ_feat)
            self.occ = rearrange(self.occ, "BT C Z H W -> BT H W Z C")
        
        # uv_feat perceiver
        with torch.no_grad():
            pos_rgb = self.position_embedding(uv_feat)
            pos_gripper = self.position_embedding(vision_gripper)
        if hasattr(self.args, 'alignment_layer') and self.args.alignment_layer == 'Linear':
            uv_feat = rearrange(uv_feat, " (B T) C BH BW ->B T (BH BW) C", B=B, T=T)
            pos_rgb = rearrange(pos_rgb, " (B T) C BH BW ->B T (BH BW) C", B=B, T=T)
            uv_feat = uv_feat + pos_rgb
            uv_feat = self.alignment_layer(uv_feat)
        elif hasattr(self.args, 'alignment_layer') and self.args.alignment_layer == 'Resampler':
            uv_feat = rearrange(uv_feat, " (B T F) C BH BW ->B T F (BH BW) C", B=B, T=T)
            pos_rgb = rearrange(pos_rgb, " (B T F) C BH BW ->B T F (BH BW) C", B=B, T=T)
            uv_feat = uv_feat + pos_rgb
            uv_feat = self.alignment_layer(uv_feat)

        # petr + perceiver
        rg_em = _calib1['rgb_gripper']['extrinsic_matrix'].clone()
        if 0:
            gripper_cam2gripper = rearrange(calib[9][:,:self.window_size], " B T H W -> (B T) H W").cpu()
            for i, _rg_em in enumerate(rg_em):
                _calib1['rgb_gripper']['extrinsic_matrix'][i]  = gripper_cam2gripper[i] # rg_em[i] @ torch.linalg.inv(state_matrix[i])
        else:
            state_matrix = rearrange(calib[6][:,:self.window_size], " B T H W -> (B T) H W").cpu()
            for i, _rg_em in enumerate(rg_em):
                _calib1['rgb_gripper']['extrinsic_matrix'][i]  = rg_em[i] @ torch.linalg.inv(state_matrix[i])
        # rs_em = _calib1['rgb_static']['extrinsic_matrix'].clone()
        # for i, _rs_em in enumerate(rs_em):
        #     _calib1['rgb_static']['extrinsic_matrix'][i]  = rs_em[i] @ torch.linalg.inv(state_matrix[i])
        pos_embed = self.petr(vision_gripper, _calib1['rgb_gripper'], pos_gripper)
        uv_gripper_feat = vision_gripper + pos_embed
        uv_gripper_feat = rearrange(uv_gripper_feat, " (B T F) C BH BW ->B T F (BH BW) C", B=B, T=T)
        uv_gripper_feat = self.perceiver(uv_gripper_feat)
        
        # send language model
        vision_x = torch.concat([uv_feat, uv_gripper_feat], dim=2)
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)
        
        feats = {}
        return vision_x, feats
        
    def loss_occ(self):
        """
        Args:
            self.preds: shape of (bs, w, h, z, c)
            self.trues: shape of (bs, w, h, z, c)
        """

        self.occ_true = self.pcd
        self.occ_true = rearrange(self.occ_true, "B T H W Z C  -> (B T) H W Z C") # B T H W Z C

        c_classes = self.occ_true.shape[-1]
        grid_cls = ['occ','r','g','b']
        loss ={}
        for ind in range(c_classes):
            preds_ind = self.occ[:,:,:,:, ind]
            trues_ind = self.occ_true[:,:,:,:, ind] 
            if ind == 0: 
                loss_ind = self.balanced_bce_loss(preds_ind, trues_ind)
            else:
                loss_ind = l1_loss(preds_ind, trues_ind, self.occ_true[:,:,:,:, 0])
            loss[f"grid_cls_{grid_cls[ind]}_loss"] = loss_ind * self.occ_loss_weight[ind]
            
        return loss
