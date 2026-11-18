import torch
import torch.nn as nn
from chartsense.model.backbone import get_backbone
from chartsense.model.heads import PatternHead, DirectionHead


class ChartSenseModel(nn.Module):
    def __init__(self, n_patterns=6, n_directions=3):
        super().__init__()
        self.backbone, in_features = get_backbone()
        self.pattern_head = PatternHead(in_features, n_patterns)
        self.direction_head = DirectionHead(in_features, n_directions)
    
    def forward(self, x):
        features = self.backbone(x)
        pattern_out = self.pattern_head(features)
        direction_out = self.direction_head(features)
        return pattern_out, direction_out


if __name__ == '__main__':
    import torch
    
    model = ChartSenseModel()
    print(model)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    pattern_out, direction_out = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Pattern output shape: {pattern_out.shape}")
    print(f"Direction output shape: {direction_out.shape}")