"""
U-Net implementation for medical image segmentation.
Architecture: standard U-Net (Ronneberger et al., 2015) with batch normalisation.
Input: (B, 1, H, W) greyscale CT/MRI slice
Output: (B, num_classes, H, W) segmentation logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 spatial dimensions
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """
    U-Net for semantic segmentation of medical images.

    Args:
        in_channels: 1 for greyscale CT/MRI
        num_classes: number of segmentation classes
        features: channel sizes for encoder stages
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 5,
                 features: list[int] = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList(
            [Down(features[i], features[i + 1]) for i in range(len(features) - 1)]
        )
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.ups = nn.ModuleList(
            [Up(features[-1] * 2 // (2 ** i), features[-1] // (2 ** i))
             for i in range(len(features))]
        )
        # Rebuild ups correctly
        up_channels = [features[-1] * 2] + [features[i] * 2 for i in range(len(features) - 1, -1, -1)]
        self.ups = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.ups.append(Up(ch, f))
            ch = f

        self.outc = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = self.bottleneck(skips[-1])

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.outc(x)


# ── Segmentation classes for brain CT/MRI ─────────────────────────────────────
BRAIN_CLASSES = {
    0: "background",
    1: "brain_parenchyma",
    2: "ventricles",
    3: "lesion",          # tumour / infarct / haemorrhage
    4: "skull",
}

ABDOMEN_CLASSES = {
    0: "background",
    1: "liver",
    2: "spleen",
    3: "kidneys",
    4: "lesion",
}

GENERIC_CLASSES = {
    0: "background",
    1: "organ",
    2: "lesion",
    3: "bone",
    4: "other",
}

CLASS_MAPS = {
    "HEAD": BRAIN_CLASSES,
    "BRAIN": BRAIN_CLASSES,
    "ABDOMEN": ABDOMEN_CLASSES,
    "GENERIC": GENERIC_CLASSES,
}

# Overlay colours (RGBA) for each class index
OVERLAY_COLORS = [
    (0, 0, 0, 0),          # background — transparent
    (0, 255, 0, 80),        # class 1 — green
    (0, 0, 255, 80),        # class 2 — blue
    (255, 0, 0, 120),       # class 3 — red (lesion)
    (255, 255, 0, 80),      # class 4 — yellow
]


def get_class_map(body_part: str) -> dict:
    part = (body_part or "").upper()
    for key in CLASS_MAPS:
        if key in part:
            return CLASS_MAPS[key]
    return GENERIC_CLASSES
