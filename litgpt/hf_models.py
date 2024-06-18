from transformers import (
    Phi3ForCausalLM, 
    Phi3Config,
    Qwen2ForCausalLM, 
    Qwen2Config
)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
print(sys.path)

from matmulfreellm.mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig

from config import Config

class Phi3(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(Phi3Config(config))

class Qwen2(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(Qwen2Config(config))

class MatMulFree(HGRNBitForCausalLM):
    def __init__(self, config):
        super().__init__(HGRNBitConfig(config))


def get_hf_models(config: Config):
    if config.name == 'phi-3':
        return Phi3(config)
    elif config.name == 'qwen2':
        return Qwen2(config)
    elif config.name == 'matmul-free-0.1B':
        return MatMulFree(config)
    else:
        raise ValueError('not impl hf models')
