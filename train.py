import torch
import os
from src.data_loader import create_data_loaders
from src.model import CatsDogsClassifier
from src.train import Trainer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model', type=str, default='resnet18')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    model = CatsDogsClassifier(num_classes=2, model_name=args.model)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs
    )
    
    train_losses, val_losses, train_accs, val_accs = trainer.train()
    
    val_loss, val_acc, cm, all_labels, all_preds = trainer.validate()
    
    trainer.plot_metrics()
    trainer.plot_confusion_matrix(cm)
    trainer.generate_classification_report(all_labels, all_preds)
    
    print(f"\n✅ Training complete!")
    print(f"🏆 Best Validation Accuracy: {trainer.best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
