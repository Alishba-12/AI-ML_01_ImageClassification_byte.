import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class CatsDogsDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        self.images = []
        self.labels = []
        
        for label, class_name in enumerate(['cat', 'dog']):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(label)
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        random.seed(42)
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images, self.labels = zip(*combined)
        
        split_idx = int(len(self.images) * train_ratio)
        if split == 'train':
            self.images = self.images[:split_idx]
            self.labels = self.labels[:split_idx]
        elif split == 'val':
            self.images = self.images[split_idx:]
            self.labels = self.labels[split_idx:]
        
        print(f"Loaded {len(self.images)} images for {split} set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(data_dir='data', batch_size=32, train_ratio=0.8):
    train_transform, val_transform = get_transforms()
    
    train_dataset = CatsDogsDataset(data_dir, transform=train_transform, 
                                   split='train', train_ratio=train_ratio)
    val_dataset = CatsDogsDataset(data_dir, transform=val_transform, 
                                 split='val', train_ratio=train_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2)
    
    return train_loader, val_loader
