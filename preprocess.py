import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class GalleryImageDataset(Dataset):
    """Custom Dataset for Gallery Image Classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(is_training=True):
    """Get data transforms for training and validation"""
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def load_dataset_paths(gallery_path, non_gallery_path):
    """Load all image paths and create labels"""
    
    image_paths = []
    labels = []
    
    # Load Gallery images (label = 1)
    gallery_path = Path(gallery_path)
    if gallery_path.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in gallery_path.glob(ext):
                image_paths.append(str(img_path))
                labels.append(1)
        print(f"Found {len([l for l in labels if l == 1])} Gallery images")
    else:
        print(f"Warning: Gallery path not found: {gallery_path}")
    
    # Load Non-Gallery images (label = 0)
    non_gallery_path = Path(non_gallery_path)
    if non_gallery_path.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in non_gallery_path.glob(ext):
                image_paths.append(str(img_path))
                labels.append(0)
        print(f"Found {len([l for l in labels if l == 0])} Non-Gallery images")
    else:
        print(f"Warning: Non-Gallery path not found: {non_gallery_path}")
    
    return image_paths, labels

def split_dataset(image_paths, labels, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """Split dataset into train, validation, and test sets"""
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    # First split: train and temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, 
        train_size=train_size, 
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: val and test
    val_ratio = val_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        train_size=val_ratio,
        random_state=random_state,
        stratify=temp_labels
    )
    
    print(f"\nDataset Split:")
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    print(f"Test set: {len(test_paths)} images")
    
    print(f"\nClass distribution in training set:")
    print(f"Gallery images: {sum(train_labels)}")
    print(f"Non-Gallery images: {len(train_labels) - sum(train_labels)}")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def create_data_loaders(train_data, val_data, test_data, batch_size=32, num_workers=4):
    """Create DataLoaders for train, validation, and test sets"""
    
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    # Create datasets
    train_dataset = GalleryImageDataset(
        train_paths, train_labels, 
        transform=get_transforms(is_training=True)
    )
    
    val_dataset = GalleryImageDataset(
        val_paths, val_labels, 
        transform=get_transforms(is_training=False)
    )
    
    test_dataset = GalleryImageDataset(
        test_paths, test_labels, 
        transform=get_transforms(is_training=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def visualize_samples(image_paths, labels, num_samples=8):
    """Visualize sample images from the dataset"""
    
    indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(indices):
            img_path = image_paths[indices[idx]]
            label = labels[indices[idx]]
            
            try:
                img = Image.open(img_path).convert('RGB')
                ax.imshow(img)
                ax.set_title(f"{'Gallery' if label == 1 else 'Non-Gallery'}")
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading image', ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    print("Sample images saved as 'sample_images.png'")
    plt.close()

if __name__ == "__main__":
    # Define paths
    GALLERY_PATH = r"C:\Users\user\Desktop\Gallery_Image_CLassiication\GalleyImage_Dataset"
    NON_GALLERY_PATH = r"C:\Users\user\Desktop\Gallery_Image_CLassiication\Non_GalleyImage_Dataset"
    
    print("=" * 60)
    print("Gallery Image Classification - Data Preprocessing")
    print("=" * 60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    image_paths, labels = load_dataset_paths(GALLERY_PATH, NON_GALLERY_PATH)
    
    if len(image_paths) == 0:
        print("Error: No images found. Please check the dataset paths.")
        exit(1)
    
    print(f"\nTotal images loaded: {len(image_paths)}")
    
    # Visualize samples
    print("\n2. Visualizing sample images...")
    visualize_samples(image_paths, labels, num_samples=8)
    
    # Split dataset
    print("\n3. Splitting dataset...")
    train_data, val_data, test_data = split_dataset(
        image_paths, labels,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )
    
    # Test data loaders
    print("\n4. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data,
        batch_size=32,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    print(f"\nData loaders created successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    print("\n5. Testing data loader...")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("Data preprocessing completed successfully!")
    print("=" * 60)