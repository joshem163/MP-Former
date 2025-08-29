import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class CNNTransformer(nn.Module):
    def __init__(self, num_classes, cnn_channels, d_model, drop_out, nhead=4, num_layers=2):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 10, 10)
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out),  # Dropout after ReLU for regularization
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_out)  # Dropout after second conv
        )

        # Flatten spatial grid into sequence of patches
        self.flatten_patches = lambda x: x.flatten(2).transpose(1, 2)  # (B, N=100, D=64)

        # Linear projection to d_model
        self.embedding = nn.Linear(cnn_channels, d_model)
        self.embedding_dropout = nn.Dropout(p=drop_out)  # Dropout after embedding

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder (with dropout inside encoder layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=drop_out  # Applies dropout to attention and feed-forward layers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout before classifier
        self.final_dropout = nn.Dropout(p=drop_out)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 4, 20, 10)
        x = self.cnn(x)  # (B, 64, 20, 10)
        x = self.flatten_patches(x)  # (B, 200, 64)
        x = self.embedding(x)  # (B, 200, d_model)
        x = self.embedding_dropout(x)  # Apply dropout to embedding

        # Add CLS token
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)  # (B, 201, d_model)

        x = self.transformer(x)  # (B, 201, d_model)
        cls_output = x[:, 0]  # (B, d_model)
        cls_output = self.final_dropout(cls_output)  # Dropout before classifier

        return self.classifier(cls_output)

#
# class CNNTransformer(nn.Module):
#     def __init__(self, num_classes, cnn_channels, d_model,drop_out, nhead=4, num_layers=2):
#         super().__init__()
#         # CNN feature extractor
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 20, 10)
#             nn.BatchNorm2d(cnn_channels),
#             nn.ReLU(),
#             nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),  # (B, 64, 20, 10)
#             nn.BatchNorm2d(cnn_channels),
#             nn.Dropout(p=drop_out),
#             nn.ReLU()
#         )
#
#         # Flatten spatial grid into sequence of patches
#         self.flatten_patches = lambda x: x.flatten(2).transpose(1, 2)  # (B, N=200, D=64)
#
#         # Linear projection to d_model
#         self.embedding = nn.Linear(cnn_channels, d_model)
#
#         # CLS token
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
#
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Classifier
#         self.classifier = nn.Linear(d_model, num_classes)
#
#     def forward(self, x):
#         # x: (B, 4, 20, 10)
#         x = self.cnn(x)  # (B, 64, 20, 10)
#         x = self.flatten_patches(x)  # (B, 200, 64)
#         x = self.embedding(x)  # (B, 200, d_model)
#
#         # Add CLS token
#         B = x.size(0)
#         cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
#         x = torch.cat((cls_token, x), dim=1)  # (B, 201, d_model)
#
#         x = self.transformer(x)  # (B, 201, d_model)
#         cls_output = x[:, 0]  # (B, d_model)
#         return self.classifier(cls_output)
