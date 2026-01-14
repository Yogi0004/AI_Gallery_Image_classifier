import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from PIL import Image
import random

# ==================== Custom Dataset (if needed for custom loading) ====================
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ==================== Data Loading & Splitting Functions ====================
def load_dataset_paths(gallery_path, non_gallery_path):
    """Collect all image paths and assign labels: 1 = Gallery, 0 = Non-Gallery"""
    image_paths = []
    labels = []

    # Gallery images (label = 1)
    for filename in os.listdir(gallery_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_paths.append(os.path.join(gallery_path, filename))
            labels.append(1)

    # Non-Gallery images (label = 0)
    for filename in os.listdir(non_gallery_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_paths.append(os.path.join(non_gallery_path, filename))
            labels.append(0)

    # Combine and shuffle
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths[:], labels[:] = zip(*combined)

    return list(image_paths), list(labels)


def split_dataset(image_paths, labels, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Split into train/val/test lists"""
    assert train_size + val_size + test_size == 1.0, "Split ratios must sum to 1.0"

    total = len(image_paths)
    train_end = int(total * train_size)
    val_end = train_end + int(total * val_size)

    # For reproducibility
    torch.manual_seed(random_state)
    indices = torch.randperm(total).tolist()

    train_data = [(image_paths[i], labels[i]) for i in indices[:train_end]]
    val_data = [(image_paths[i], labels[i]) for i in indices[train_end:val_end]]
    test_data = [(image_paths[i], labels[i]) for i in indices[val_end:]]

    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, batch_size=4, num_workers=0):
    """Create DataLoaders with proper transforms"""

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Only resize + normalize for val/test
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset(
        image_paths=[x[0] for x in train_data],
        labels=[x[1] for x in train_data],
        transform=train_transform
    )

    val_dataset = CustomImageDataset(
        image_paths=[x[0] for x in val_data],
        labels=[x[1] for x in val_data],
        transform=val_test_transform
    )

    test_dataset = CustomImageDataset(
        image_paths=[x[0] for x in test_data],
        labels=[x[1] for x in test_data],
        transform=val_test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# ==================== Model Definition ====================
class GalleryImageClassifier(nn.Module):
    """ResNet-18 based Gallery Image Classifier"""
    
    def __init__(self, num_classes=2):
        super(GalleryImageClassifier, self).__init__()
        
        # Use modern weights (pretrained=True is deprecated)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        num_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


# ==================== Training Function ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, device='cuda', save_path='models'):
    os.makedirs(save_path, exist_ok=True)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rates': []
    }
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train' and (batch_idx + 1) % 10 == 0:
                    print(f'Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                if scheduler is not None:
                    scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'history': history
                }, os.path.join(save_path, 'best_model.pth'))
                
                print(f'✓ Best model saved! Validation Accuracy: {best_acc:.4f}')
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch completed in {epoch_time//60:.0f}m {epoch_time%60:.0f}s')
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n" + "=" * 60)
    print(f'Training completed!')
    print(f'Best Validation Accuracy: {best_acc:.4f}')
    print("=" * 60)
    
    model.load_state_dict(best_model_wts)
    return model, history


# ==================== Evaluation & Plotting Functions (unchanged) ====================
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Gallery', 'Gallery']))
    
    return all_preds, all_labels, all_probs


def plot_training_history(history, save_path='models'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['learning_rates'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary:
    
    Best Train Acc: {max(history['train_acc']):.4f}
    Best Val Acc: {max(history['val_acc']):.4f}
    
    Final Train Loss: {history['train_loss'][-1]:.4f}
    Final Val Loss: {history['val_loss'][-1]:.4f}
    
    Final Train Acc: {history['train_acc'][-1]:.4f}
    Final Val Acc: {history['val_acc'][-1]:.4f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to {save_path}/training_history.png")
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, save_path='models'):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Gallery', 'Gallery'],
                yticklabels=['Non-Gallery', 'Gallery'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}/confusion_matrix.png")
    plt.close()


def plot_roc_curve(all_labels, all_probs, save_path='models'):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}/roc_curve.png")
    plt.close()


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Configuration
    GALLERY_PATH = r"C:\Users\user\Desktop\Gallery_Image_CLassiication\GalleyImage_Dataset"
    NON_GALLERY_PATH = r"C:\Users\user\Desktop\Gallery_Image_CLassiication\Non_GalleyImage_Dataset"
    SAVE_PATH = 'models'
    
    # Hyperparameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\n1. Loading dataset...")
    image_paths, labels = load_dataset_paths(GALLERY_PATH, NON_GALLERY_PATH)
    
    if len(image_paths) == 0:
        print("Error: No images found. Please check the dataset paths.")
        exit(1)
    
    print(f"Total images: {len(image_paths)}")
    
    print("\n2. Splitting dataset...")
    train_data, val_data, test_data = split_dataset(
        image_paths, labels,
        train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
    )
    
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    print("\n4. Initializing model...")
    model = GalleryImageClassifier(num_classes=2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    print("\n5. Training model...")
    model, history = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_path=SAVE_PATH
    )
    
    print("\n6. Plotting training history...")
    plot_training_history(history, save_path=SAVE_PATH)
    
    print("\n7. Evaluating on test set...")
    all_preds, all_labels, all_probs = evaluate_model(model, test_loader, device=device)
    
    print("\n8. Generating evaluation plots...")
    plot_confusion_matrix(all_labels, all_preds, save_path=SAVE_PATH)
    plot_roc_curve(all_labels, all_probs, save_path=SAVE_PATH)
    
    print("\n9. Saving final model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, os.path.join(SAVE_PATH, 'final_model.pth'))
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print(f"Models and plots saved in '{SAVE_PATH}' directory")
    print("=" * 60)