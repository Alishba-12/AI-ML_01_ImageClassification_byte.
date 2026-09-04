import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime
from src.model import save_model

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 lr=0.001, epochs=20):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                          patience=3, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss/(total/self.train_loader.batch_size):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validating')
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({
                    'Loss': f'{total_loss/(total/self.val_loader.batch_size):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        val_acc = 100. * correct / total
        cm = confusion_matrix(all_labels, all_preds)
        
        return total_loss / len(self.val_loader), val_acc, cm, all_labels, all_preds
    
    def train(self):
        print(f"Training on {self.device}")
        print(f"Total epochs: {self.epochs}")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_cm, _, _ = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_path = f'models/best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                save_model(self.model, save_path)
                print(f"✓ New best model saved! Acc: {val_acc:.2f}%")
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        print(f"\n🏆 Best Validation Accuracy: {self.best_val_acc:.2f}%")
        return self.train_losses, self.val_losses, self.train_accs, self.val_accs
    
    def plot_metrics(self, save_path='outputs/metrics/'):
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_metrics.png'), dpi=300)
        plt.show()
        print(f"Metrics plots saved to {save_path}")
    
    def plot_confusion_matrix(self, cm, save_path='outputs/confusion_matrix/'):
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Cats vs Dogs')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300)
        plt.show()
        print(f"Confusion matrix saved to {save_path}")
    
    def generate_classification_report(self, all_labels, all_preds, 
                                      save_path='outputs/metrics/'):
        os.makedirs(save_path, exist_ok=True)
        report = classification_report(all_labels, all_preds, 
                                      target_names=['Cat', 'Dog'])
        
        with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
            f.write("Classification Report - Cats vs Dogs\n")
            f.write("=" * 40 + "\n\n")
            f.write(report)
        
        print("\nClassification Report:")
        print(report)
        return report
