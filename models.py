import timm
import torch
from torch import nn

# --- Models --- #
class ViT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0, 
            in_chans=3, drop_path_rate=0.2
        )
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(self.backbone.num_features, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.dropout(self.backbone.forward_features(x))[:, 0])


class DenseNet(nn.Module):
    def __init__(self, heads=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "densenet169", pretrained=pretrained, global_pool="avg", 
            num_classes=0, drop_rate=0.1
        )
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(1664, heads))

    def forward(self, x):
        return self.head(self.backbone(x))


# --- Losses --- #
class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets,
           pos_weight=self.pos_weight, reduction="none"
        )
        prob = torch.sigmoid(logits)
        pt = targets * prob + (1 - targets) * (1 - prob)
        return (((1 - pt) ** 2) * bce).mean()


class BranchLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        losses = []
        for head in range(2):
            active = targets[:, head] >= 0
            if active.any():
                losses.append(FocalLoss(self.pos_weight[head])(
                    logits[active, head], targets[active, head]))
        return torch.stack(losses).mean()
