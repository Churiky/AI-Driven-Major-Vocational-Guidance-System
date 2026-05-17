import torch
import torch.nn as nn

class CareerTransformer(nn.Module):

    def __init__(self,input_dim,num_classes):

        super().__init__()

        self.embedding=nn.Linear(input_dim,128)

        encoder_layer=nn.TransformerEncoderLayer(

            d_model=128,
            nhead=8,
            dim_feedforward=256,
            batch_first=True

        )

        self.transformer=nn.TransformerEncoder(

            encoder_layer,
            num_layers=3

        )

        self.fc=nn.Sequential(

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,num_classes)

        )

    def forward(self,x):

        x=self.embedding(x)

        x=x.unsqueeze(1)

        x=self.transformer(x)

        x=x.squeeze(1)

        out=self.fc(x)

        return out


class FeatureTokenizerTransformer(nn.Module):

    def __init__(
        self,
        num_features,
        num_classes,
        d_model=64,
        nhead=4,
        num_layers=3,
        ff_multiplier=3,
        dropout=0.15,
    ):

        super().__init__()

        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):

        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.feature_embedding

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        encoded = self.transformer(torch.cat([cls_token, tokens], dim=1))

        return self.classifier(encoded[:, 0, :])

class MultimodalCareerTransformer(nn.Module):
    def __init__(
        self,
        num_acad_features: int,
        num_psych_features: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        
        # Riêng rẽ embedding cho 2 luồng
        self.acad_embedding = nn.Parameter(torch.randn(1, num_acad_features, d_model) * 0.02)
        self.psych_embedding = nn.Parameter(torch.randn(1, num_psych_features, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Encoder layer cho luồng Academic và Psych (để học ngữ cảnh nội bộ của từng luồng)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.acad_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers//2 or 1)
        self.psych_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers//2 or 1)
        
        # Cross-Attention layer: Academic "query" thông tin từ Psychological
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.cross_attention = nn.TransformerDecoder(decoder_layer, num_layers=num_layers//2 or 1)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        
        self.num_acad = num_acad_features
        self.num_psych = num_psych_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x có shape: (batch_size, num_acad + num_psych)
        acad_x = x[:, :self.num_acad]
        psych_x = x[:, self.num_acad:]
        
        # Chiếu giá trị 1D -> d_model
        acad_tokens = self.value_projection(acad_x.unsqueeze(-1)) + self.acad_embedding
        psych_tokens = self.value_projection(psych_x.unsqueeze(-1)) + self.psych_embedding
        
        # Encoder cục bộ
        acad_encoded = self.acad_encoder(acad_tokens)
        psych_encoded = self.psych_encoder(psych_tokens)
        
        # Thêm CLS token vào acad_encoded để làm query chính
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        query_seq = torch.cat([cls_token, acad_encoded], dim=1)
        
        # Cross Attention: query = query_seq, memory = psych_encoded
        cross_out = self.cross_attention(tgt=query_seq, memory=psych_encoded)
        
        # Lấy output tại vị trí CLS (vị trí 0)
        return self.classifier(cross_out[:, 0, :])


