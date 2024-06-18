def get_config(model_name):
    if model_name not in name_to_config:
        raise ValueError('model not impl')
    conf_dict = name_to_config[model_name]
    return conf_dict

configs = []


################
# qwen2
################
qwen2 = [
    dict(
        name="qwen2-0.5B",
        hf_config=dict(org="qwen", name="qwen2-0.5B"),
        vocab_size=50257,
        attention_dropout=0.0,
        bos_token_id=1, ## llm-jp
        eos_token_id=7, ## llm-jp
        hidden_act="silu",
        hidden_size=896,
        initializer_range=0.02,
        intermediate_size=4864,
        max_position_embeddings=131072,
        max_window_layers=24,
        model_type="qwen2",
        num_attention_heads=14,
        num_hidden_layers=24,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=131072,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        transformers_version="4.40.1",
        use_cache=True,
        use_sliding_window=False,
    ),
    dict(
        name="qwen2-0.1B",
        hf_config=dict(org="qwen", name="qwen2-0.1B"),
        vocab_size=50257,
        attention_dropout=0.0,
        bos_token_id=1, ## llm-jp
        eos_token_id=7, ## llm-jp
        hidden_act="silu",
        hidden_size=512,
        initializer_range=0.02,
        intermediate_size=2048,
        max_position_embeddings=131072,
        max_window_layers=24,
        model_type="qwen2",
        num_attention_heads=6,
        num_hidden_layers=6,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        sliding_window=131072,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
        transformers_version="4.40.1",
        use_cache=True,
        use_sliding_window=False,
    ),
]
configs.extend(qwen2)

################
# matmul free
################
matmul_free = [
    dict(
        name="matmul-free-0.1B",
        hf_config=dict(org="matmul-free", name="matmul-free-0.1B"),
        vocab_size=50257,
        hidden_size = 896,
        num_hidden_layers = 8,
        attn_mode = "fused_recurrent",
        num_heads = 1,
        expand_ratio = 1,
        use_short_conv = False,
        conv_size = 4,
        share_conv_kernel = True,
        use_lower_bound = True,
        hidden_ratio = 4,
        intermediate_size = None,
        hidden_act = "swish",
        max_position_embeddings = 1024,
        rms_norm_eps = 1e-6,
        use_cache = True,
        pad_token_id = None,
        bos_token_id = 1,  ## llm-jp
        eos_token_id = 7,  ## llm-jp
        tie_word_embeddings = False,
        initializer_range = 0.02,
        fuse_cross_entropy = True,
        max_seq_length=1024,
    )
]

configs.extend(matmul_free)

name_to_config = {config["name"]: config for config in configs}