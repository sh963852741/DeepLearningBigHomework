import torch
import torch.nn
import math
class PUEForecast(torch.nn.Module):
    def __init__(self):
        super(PUEForecast, self).__init__()
        self.d_model = 128
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dropout=0.1,
            dim_feedforward=8 * self.d_model,
        )
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=4,
            dropout=0.1,
            dim_feedforward=8 * self.d_model,
        )

        self.pos_encoder = PositionalEncoding(self.d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.linear = torch.nn.Linear(self.d_model, 1)


    def forward(self, X, target_in):
        # X, shape:[batch_size, 多少行, 每行多少特征]
        X = self.pos_encoder(X)
        X = self.encoder(src=X) # * math.sqrt(self.d_model)
        
        target_in = self.pos_encoder(target_in)
        mask = gen_trg_mask(target_in.size(0))
        output = self.decoder(tgt=target_in, memory=X, tgt_mask=mask)
        output = self.linear(output)
        return output

def gen_trg_mask(length):
    mask = torch.tril(torch.ones(length, length)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)