import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from chartsense.data.dataset import ChartDataset, PATTERNS, DIRECTIONS
from chartsense.model.chartsense import ChartSenseModel

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(model_path='models/best_model.pth'):
    model = ChartSenseModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def get_predictions(model, loader):
    all_pattern_preds = []
    all_pattern_labels = []
    all_direction_preds = []
    all_direction_labels = []

    with torch.no_grad():
        for images, pattern_labels, direction_labels in loader:
            pattern_out, direction_out = model(images)

            all_pattern_preds.extend(pattern_out.argmax(1).numpy())
            all_pattern_labels.extend(pattern_labels.numpy())
            all_direction_preds.extend(direction_out.argmax(1).numpy())
            all_direction_labels.extend(direction_labels.numpy())

    return (np.array(all_pattern_preds), np.array(all_pattern_labels),
            np.array(all_direction_preds), np.array(all_direction_labels))


def plot_confusion_matrix(labels, preds, class_names, title, save_path):
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1
    )
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def print_and_save_report(labels, preds, class_names, title, save_path):
    report = classification_report(labels, preds, target_names=class_names)
    print(f"\n{title}")
    print("=" * 60)
    print(report)
    
    with open(save_path, 'w') as f:
        f.write(f"{title}\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"Saved: {save_path}")


def evaluate(model_path='models/best_model.pth'):
    Path('evaluation').mkdir(exist_ok=True)
    
    dataset = ChartDataset(
        labels_csv='data/labels.csv',
        images_dir='data/synthetic',
        transform=val_transforms
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print("Loading model...")
    model = load_model(model_path)
    
    print(f"Running inference on {len(dataset)} samples...")
    pattern_preds, pattern_labels, direction_preds, direction_labels = get_predictions(model, loader)
    
    # Pattern evaluation
    pattern_acc = (pattern_preds == pattern_labels).mean()
    print(f"\nPattern Accuracy: {pattern_acc:.4f}")
    
    plot_confusion_matrix(
        pattern_labels, pattern_preds,
        class_names=PATTERNS,
        title='Pattern Classification Confusion Matrix',
        save_path='evaluation/pattern_confusion_matrix.png'
    )
    
    print_and_save_report(
        pattern_labels, pattern_preds,
        class_names=PATTERNS,
        title='Pattern Classification Report',
        save_path='evaluation/pattern_report.txt'
    )
    
    # Direction evaluation
    direction_acc = (direction_preds == direction_labels).mean()
    print(f"\nDirection Accuracy: {direction_acc:.4f}")
    
    plot_confusion_matrix(
        direction_labels, direction_preds,
        class_names=DIRECTIONS,
        title='Direction Prediction Confusion Matrix',
        save_path='evaluation/direction_confusion_matrix.png'
    )
    
    print_and_save_report(
        direction_labels, direction_preds,
        class_names=DIRECTIONS,
        title='Direction Prediction Report',
        save_path='evaluation/direction_report.txt'
    )
    
    print("\nEvaluation complete. Results saved to evaluation/")


if __name__ == '__main__':
    evaluate()