import timm
import torch
import torch.nn as nn


class ProtoNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)

    def embed(self, x):
        return self.encoder(x)

    def forward(self, support_x, support_y, query_x):
        zs = self.embed(support_x)
        zq = self.embed(query_x)
        classes = torch.unique(support_y)
        protos = torch.stack([zs[support_y == c].mean(0) for c in classes])
        # negative squared euclidean distance as logits, as in Snell et al. 2017
        return -torch.cdist(zq, protos) ** 2
