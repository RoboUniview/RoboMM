from logging import debug
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import open_clip
from typing import Optional
# from robouniview.models.flamingo_bc import BCFlamingo
from robouniview.models.flamingo_mpt import MPTFlamingo
from robouniview.models.flamingo_keyframe import MPTFlamingo as KeyframeFlamingo
#from robouniview.models.robouniview_mpt import MPTFlamingo
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from open_flamingo.src.factory import _infer_decoder_layers_attr_name

from .flamingo_mask import FlamingoLMMaskMixin, GPTBlockMaskMixin, MultiheadAttentionMaskMixin

mpt_dict = {
    "mpt_3b": {
        "lang_encoder_path": "path_to/mpt-1b-redpajama-200b", 
        "tokenizer_path": "path_to/mpt-1b-redpajama-200b", 
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
    }, 
    "mpt_dolly_3b": {
        "lang_encoder_path": "/project/robotic/RoboFlamingo/checkpoints/mpt-1b-redpajama-200b-dolly",
        "tokenizer_path": "/project/robotic/RoboFlamingo/checkpoints/mpt-1b-redpajama-200b-dolly",
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "/project/robotic/RoboFlamingo/checkpoints/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt"
    },
    "keyframe_3b": {
        "lang_encoder_path": "/project/robotic/RoboFlamingo/checkpoints/mpt-1b-redpajama-200b-dolly",
        "tokenizer_path": "/project/robotic/RoboFlamingo/checkpoints/mpt-1b-redpajama-200b-dolly",
        "cross_attn_every_n_layers": 1,
        "openflamingo_checkpoint": "/project/robotic/RoboFlamingo/checkpoints/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt"
    },
    "mpt_4b": {
        "lang_encoder_path": "path_to/RedPajama-INCITE-Instruct-3B-v1", 
        "tokenizer_path": "path_to/RedPajama-INCITE-Instruct-3B-v1", 
        "cross_attn_every_n_layers": 2,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-4B-vitl-rpj3b-langinstruct/checkpoint.pt"
    },
    "mpt_base_4b": {
        "lang_encoder_path": "path_to/RedPajama-INCITE-Base-3B-v1", 
        "tokenizer_path": "path_to/RedPajama-INCITE-Base-3B-v1", 
        "cross_attn_every_n_layers": 2,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-4B-vitl-rpj3b/checkpoint.pt"
    },
    "mpt_9b": {
        "lang_encoder_path": "path_to/mpt-7b", 
        "tokenizer_path": "path_to/mpt-7b", 
        "cross_attn_every_n_layers": 4,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"
    },
    "llama_9b": {
        "lang_encoder_path": "path_to/llama-7b-hf-jxu124", 
        "tokenizer_path": "path_to/llama-7b-hf-jxu124", 
        "cross_attn_every_n_layers": 4,
        "openflamingo_checkpoint": "path_to/OpenFlamingo-9B/checkpoint.pt"
    }
}



def get_transforms(
    clip_vision_encoder_path: str = "ViT-L-14",
    clip_vision_encoder_pretrained: str = "openai",
    tokenizer_path: str = "path_to/llama-7b-hf-jxu124",
    use_local_files: bool = False,
):
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )

    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    return image_processor, text_tokenizer


def create_model_and_transforms(
    args,
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    # this is the window size sampled from the episode
    window_size: int = 32,
    freeze_embed: bool = False,
    train_params = -1,
    use_gripper=False,
    use_state=False,
    last_action=False,
    fusion_mode='',
    pad_length=-1,
    debug=False,
    sep_resampler=False,
    sep_lm_head=False,
    unfreeze_vit=False,
    return_feature=False,
    multi_step_action=1,
    llm_name='llama_9b',
    pooling='max',
    residual=False,
    tcp_rel=False,
    replan=-1,
    decoder_type='lstm',
    hidden_size=None,
    freeze_sampler=False,
    fwd_pred=False, 
    fwd_pred_hand=False,
    no_image_patch=False,
    global_latent=1,
    refresh=-1,
    clip_cache_dir=None,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained,cache_dir=clip_cache_dir,
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=use_local_files)
    # add Flamingo special tokens to the tokenizer
    # preimg0_token_id = tokenizer("<static0>", add_special_tokens=False)["input_ids"][-1]
    # preimg1_token_id = tokenizer("<static1>", add_special_tokens=False)["input_ids"][-1]
    # preimg2_token_id = tokenizer("<static2>", add_special_tokens=False)["input_ids"][-1]
    # preimg3_token_id = tokenizer("<static3>", add_special_tokens=False)["input_ids"][-1]
    # preimg4_token_id = tokenizer("<static4>", add_special_tokens=False)["input_ids"][-1]
    # preimg5_token_id = tokenizer("<static5>", add_special_tokens=False)["input_ids"][-1]
    # preimg6_token_id = tokenizer("<static6>", add_special_tokens=False)["input_ids"][-1]
    # preimg7_token_id = tokenizer("<static7>", add_special_tokens=False)["input_ids"][-1]
    # preimg0_token_id = tokenizer("<gripper0>", add_special_tokens=False)["input_ids"][-1]
    # preimg1_token_id = tokenizer("<gripper1>", add_special_tokens=False)["input_ids"][-1]
    # preimg2_token_id = tokenizer("<gripper2>", add_special_tokens=False)["input_ids"][-1]
    # preimg3_token_id = tokenizer("<gripper3>", add_special_tokens=False)["input_ids"][-1]
    # preimg4_token_id = tokenizer("<gripper4>", add_special_tokens=False)["input_ids"][-1]
    # preimg5_token_id = tokenizer("<gripper5>", add_special_tokens=False)["input_ids"][-1]
    # preimg6_token_id = tokenizer("<gripper6>", add_special_tokens=False)["input_ids"][-1]
    # preimg7_token_id = tokenizer("<gripper7>", add_special_tokens=False)["input_ids"][-1]
    # preimg0_token_id = tokenizer("<obs0>", add_special_tokens=False)["input_ids"][-1]
    # preimg1_token_id = tokenizer("<obs1>", add_special_tokens=False)["input_ids"][-1]
    # preimg2_token_id = tokenizer("<obs2>", add_special_tokens=False)["input_ids"][-1]
    # preimg3_token_id = tokenizer("<obs3>", add_special_tokens=False)["input_ids"][-1]
    # preimg4_token_id = tokenizer("<obs4>", add_special_tokens=False)["input_ids"][-1]
    # preimg5_token_id = tokenizer("<obs5>", add_special_tokens=False)["input_ids"][-1]
    # preimg6_token_id = tokenizer("<obs6>", add_special_tokens=False)["input_ids"][-1]
    # preimg7_token_id = tokenizer("<obs7>", add_special_tokens=False)["input_ids"][-1]
    if args.action_token:
        print("YF: add start and end token!!!!!!!!!!!!!!!!!")
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>", "<action>", "<static0>", "<static1>","<static2>","<static3>","<static4>","<static5>","<static6>","<static7>","<static_s>","<static_e>"]}
        )
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<gripper0>", "<gripper1>","<gripper2>","<gripper3>","<gripper4>","<gripper5>","<gripper6>","<gripper7>","<gripper_s>","<gripper_e>"]}
        )
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<obs0>", "<obs1>","<obs2>","<obs3>","<obs4>","<obs5>","<obs6>","<obs7>","<obs_s>","<obs_e>"]}
        )
    else:
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if debug:
        # Load the local checkpoint into a model instance.
        lang_encoder = AutoModelForCausalLM.from_pretrained(lang_encoder_path, ignore_keys=["config"], trust_remote_code=True)
        # Set the `init_weights` parameter to `False` to prevent the model from loading the pretrained weights.
        lang_encoder.init_weights(False)
    else:
        print("language_encoder_path:{}".format(lang_encoder_path)) # yiyang_mod
        lang_encoder = AutoModelForCausalLM.from_pretrained(lang_encoder_path, local_files_only=use_local_files, trust_remote_code=True) # , attn_impl="triton")
        # print(lang_encoder_path)
        # if llm_name == 'llama':
        #     lang_encoder = AutoModelForCausalLM.from_pretrained(
        #     lang_encoder_path, local_files_only=use_local_files
        # )
        # else:
        #     # name = 'mosaicml/mpt-7b'
        #     config = {
        #         "model_type": "auto",
        #         "add_lm_head": True,
        #     }
        #     lang_encoder = AutoModelForCausalLM.from_pretrained(
        #         lang_encoder_path, local_files_only=use_local_files
        #     )
    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    # 模型没有get embedding方法，单独加一个，返回embedding
    if "mpt-1b-redpajama-200b" in lang_encoder_path:
        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte
            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings
        extend_instance(lang_encoder, EmbeddingFnMixin)
    extend_instance(lang_encoder, FlamingoLMMixin)
    # extend_instance(lang_encoder, FlamingoLMMaskMixin) # YF: add 修改forward_super的传参
    # for b_idx, block in enumerate(lang_encoder.transformer.blocks):  # YF: add
    #     extend_instance(block, GPTBlockMaskMixin)
    #     extend_instance(block.attn, MultiheadAttentionMaskMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    # print(lang_encoder.base_model_prefix)
    # print(getattr(lang_encoder, lang_encoder.base_model_prefix, lang_encoder))
    # print(lang_encoder)
    lang_encoder.resize_token_embeddings(len(text_tokenizer)) #yiyang_question
    
    if 'llama' in llm_name: Model_fn = BCFlamingo
    elif 'mpt' in llm_name: Model_fn = MPTFlamingo
    elif'keyframe' in llm_name: Model_fn = KeyframeFlamingo
    else: raise NotImplementedError
    
    model = Model_fn(
        args,
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        window_size=window_size,
        use_gripper=use_gripper,
        use_state=use_state,
        fusion_mode=fusion_mode,
        last_action=last_action,
        pad_length=pad_length,
        sep_resampler=sep_resampler,
        sep_lm_head=sep_lm_head,
        return_feature=return_feature,
        multi_step_action=multi_step_action,
        llm=llm_name,
        pooling=pooling,
        residual=residual,
        tcp_rel=tcp_rel,
        replan=replan,
        decoder_type=decoder_type,
        hidden_size=hidden_size,
        refresh=refresh,
        fwd_pred=fwd_pred,
        fwd_pred_hand=fwd_pred_hand,
        no_image_patch=no_image_patch,
        global_latent=global_latent,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    # model.perceiver.requires_grad_(True)
    if args.train_uvformer:
        if hasattr(model, 'uvformer'):              model.uvformer.requires_grad_(True) ###uvformer_lff
        if hasattr(model, 'uvformer'):              model.uvformer.requires_grad_(True) ###uvformer_lff
        if hasattr(model, 'uvformer_gripper'):      model.uvformer_gripper.requires_grad_(True) ###uvformer_lff
        if hasattr(model, 'occ_decoder_UVFormer'):  model.occ_decoder_UVFormer.requires_grad_(True) ###uvformer_lff
        if hasattr(model, 'Upsample2d_3d_UVFormer'):model.Upsample2d_3d_UVFormer.requires_grad_(True) ###uvformer_lff
    if hasattr(args, 'train_uvformer_gripper') and args.train_uvformer_gripper:
        if hasattr(model, 'uvformer_gripper'):      model.uvformer_gripper.requires_grad_(True) ###uvformer_lff
    if hasattr(model, 'linear'):                    model.linear.requires_grad_(True) ###uvformer_lff
    if hasattr(model, 'alignment_layer'):           model.alignment_layer.requires_grad_(True) ###uvformer_lff
    if hasattr(model, 'alignment_layer_gripper'):   model.alignment_layer_gripper.requires_grad_(True)
    if hasattr(model, 'petr'):                      model.petr.requires_grad_(True)

    if hasattr(model, 'vlm'): 
        # model.vlm.requires_grad_(True)
        # model.vlm.gated_cross_attn_layers.requires_grad_(True)
        model.vlm.get_input_embeddings().requires_grad_(True) # 需要更新embedding
        model.vlm_perceiver.requires_grad_(True) # 需要更新视觉特征到语言特征的映射

    if "UVFormer" not in args.fusion_mode: # 注意此处，当不使用UVFormer的时候，occ_decoder和Upsample2d_3d必须训练
        model.rgb.requires_grad_(True)
        model.gripper.requires_grad_(True)
    # decoder image
    model.decoder_embed.requires_grad_(True)
    model.decoder_blocks.requires_grad_(True)
    model.decoder_norm.requires_grad_(True)
    model.decoder_pred.requires_grad_(True)
    # decoder obj/occ
    if hasattr(model, 'decoder_embed_obs'):     model.decoder_embed_obs.requires_grad_(True)
    if hasattr(model, 'decoder_blocks_obs'):    model.decoder_blocks_obs.requires_grad_(True)
    if hasattr(model, 'occ_decoder'):           model.occ_decoder.requires_grad_(True) 
    if hasattr(model, 'Upsample2d_3d'):         model.Upsample2d_3d.requires_grad_(True)

    if hasattr(model, 'T_Embedding_layer'): model.T_Embedding_layer.requires_grad_(True)
    if hasattr(args, 'unfreeze_old_decoder_blocks') and args.unfreeze_old_decoder_blocks:
        model.lang_encoder.requires_grad_(True)
    
    if train_params == -1:
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        model.perceiver.requires_grad_(True)
    else:
        param_per_layer = 140
        layer_num = int(train_params / param_per_layer + 0.5)
        cnt = 0
        for ix in range(len(model.lang_encoder.gated_cross_attn_layers)-1, -1, -1):
            if cnt >= layer_num:
                break
            if model.lang_encoder.gated_cross_attn_layers[ix] is not None:
                model.lang_encoder.gated_cross_attn_layers[ix].requires_grad_(True)
                cnt += 1
    if freeze_sampler:
        model.perceiver.requires_grad_(False)
    if not freeze_embed:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
    model.lang_encoder.lm_head.requires_grad_(True)

    if model.sep_lm_head:
        model.lm_head.requires_grad_(True)
    if model.use_diff:
        model.diffusion_model.requires_grad_(True)
    if unfreeze_vit:
        assert False # 后边forward的时候不更新梯度
        model.vision_encoder.requires_grad_(True)
    # # Unfreeze the action head 
    # model.action_head.requires_grad_(True)

    print(f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

    return model, image_processor, text_tokenizer
