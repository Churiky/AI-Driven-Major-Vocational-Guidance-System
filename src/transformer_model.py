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
