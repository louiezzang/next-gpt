from typing import Optional

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from ..base import Actor


class GPTActor(Actor):
    """
    GPT Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (GPT2Config): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): Rank of the LoRa layer. 0 means LoRA is not applied. Defaults to 0.
        lora_train_bias (str): Bias training strategy for the LoRa layer.
            'none' means it doesn't train biases. 'all' means it trains all biases. 'lora_only' means it only trains biases of LoRA layers.
            Defaults to 'none'.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = GPT2LMHeadModel.from_pretrained(pretrained)
        elif config is not None:
            model = GPT2LMHeadModel(config)
        else:
            model = GPT2LMHeadModel(GPT2Config())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model, lora_rank, lora_train_bias)
