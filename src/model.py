"""
model.py
--------
Defines the image encoder, text encoder, and three classifiers used for
comparison:
  1. ImageOnlyClassifier  - unimodal baseline (CNN only)
  2. TextOnlyClassifier   - unimodal baseline (TF-IDF + MLP only)
  3. FusionClassifier     - intermediate (joint-representation) fusion:
                            image embedding + text embedding are
                            concatenated and passed through a shared
                            classification head, trained end-to-end.
"""
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Small CNN feature extractor -> fixed-length embedding."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 16 -> 8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class TextEncoder(nn.Module):
    """Projects a fixed TF-IDF vector into a dense embedding."""

    def __init__(self, tfidf_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tfidf_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ImageOnlyClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=64):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, image, text_vec=None):
        z = self.image_encoder(image)
        return self.head(z)


class TextOnlyClassifier(nn.Module):
    def __init__(self, num_classes, tfidf_dim, embed_dim=64):
        super().__init__()
        self.text_encoder = TextEncoder(tfidf_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, image=None, text_vec=None):
        z = self.text_encoder(text_vec)
        return self.head(z)


class FusionClassifier(nn.Module):
    """Intermediate fusion: concatenate image + text embeddings, then classify."""

    def __init__(self, num_classes, tfidf_dim, embed_dim=64):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(tfidf_dim, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, text_vec):
        img_z = self.image_encoder(image)
        txt_z = self.text_encoder(text_vec)
        fused = torch.cat([img_z, txt_z], dim=1)
        return self.classifier(fused)
