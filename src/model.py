import torch
import torch.nn as nn
from torchvision import models
import os

class CatsDogsClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name='resnet18', pretrained=True):
        super(CatsDogsClassifier, self).__init__()
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    def forward(self, x):
        return self.backbone(x)

def save_model(model, path='models/model.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': 2,
        'model_name': 'resnet18'
    }, path)
    print(f"Model saved to {path}")

def load_model(path='models/model.pth', device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model = CatsDogsClassifier(
        num_classes=checkpoint['num_classes'], 
        model_name=checkpoint.get('model_name', 'resnet18')
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)
