import torch.nn as nn
from torch import Tensor


class TranslationModel(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        dim_feedforward: int,
        n_head: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dropout_prob: float,
    ):
        """
        Creates a standard Transformer encoder-decoder model.
        :param num_encoder_layers: Number of encoder layers
        :param num_decoder_layers: Number of decoder layers
        :param emb_size: Size of intermediate vector representations for each token
        :param dim_feedforward: Size of intermediate representations in FFN layers
        :param n_head: Number of attention heads for each layer
        :param src_vocab_size: Number of tokens in the source language vocabulary
        :param tgt_vocab_size: Number of tokens in the target language vocabulary
        :param dropout_prob: Dropout probability throughout the model
        """
        super().__init__()
        # your code here

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        """
        Given tokens from a batch of source and target sentences, predict logits for next tokens in target sentences.
        """
        pass
