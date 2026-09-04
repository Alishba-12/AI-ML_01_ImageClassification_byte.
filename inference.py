import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.model import load_model
import argparse

class InferenceEngine:
    def __init__(self, model_path='models/best_model.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path, self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = ['Cat', 'Dog']
    
    def predict_single(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, image
    
    def visualize_predictions(self, image_paths, ground_truth=None, 
                            save_path='outputs/sample_predictions/'):
        os.makedirs(save_path, exist_ok=True)
        
        results = []
        for path in image_paths:
            pred_class, confidence, img = self.predict_single(path)
            results.append({
                'path': path,
                'predicted_class': pred_class,
                'class_name': self.class_names[pred_class],
                'confidence': confidence,
                'image': img
            })
        
        n_images = len(results)
        cols = 5
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        axes = axes.flatten() if rows > 1 else [axes]
        
        for idx, result in enumerate(results):
            ax = axes[idx]
            ax.imshow(result['image'])
            
            color = 'green'
            title = f"{result['class_name']}\n({result['confidence']:.2%})"
            ax.set_title(title, color=color, fontsize=12)
            ax.axis('off')
        
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'predictions.png'), dpi=300)
        plt.show()
        print(f"Predictions saved to {save_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    parser.add_argument('--image_dir', type=str, help='Directory with images')
    parser.add_argument('--image_paths', nargs='+', help='List of image paths')
    parser.add_argument('--n_samples', type=int, default=10)
    args = parser.parse_args()
    
    if args.image_paths:
        image_paths = args.image_paths
    elif args.image_dir:
        image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = image_paths[:args.n_samples]
    else:
        data_dir = 'data'
        image_paths = []
        for class_name in ['cat', 'dog']:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                imgs = [os.path.join(class_dir, f) for f in os.listdir(class_dir)[:5]]
                image_paths.extend(imgs)
        image_paths = image_paths[:args.n_samples]
    
    if not image_paths:
        print("No images found!")
        return
    
    print(f"Running inference on {len(image_paths)} images...")
    
    engine = InferenceEngine(args.model_path)
    results = engine.visualize_predictions(image_paths)

if __name__ == "__main__":
    main()
