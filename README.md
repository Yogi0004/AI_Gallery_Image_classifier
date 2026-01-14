# 🏗️ Gallery Image Detector

A deep learning-based system to automatically detect and classify Gallery images using ResNet-18 architecture.

## 📋 Overview

This project uses a Convolutional Neural Network (CNN) to classify images as either **Gallery images** or **Non-Gallery images** with high accuracy and confidence scores.

### What is a Gallery Image?

A Gallery image refers to an individual photo within a collection (gallery), displayed on websites, apps, or phones as part of an organized visual showcase, often in grids or slideshows.

## 🚀 Features

- ✅ Binary classification (Gallery vs Non-Gallery)
- ✅ ResNet-18 based deep learning model
- ✅ Interactive web interface using Gradio
- ✅ Real-time prediction with confidence scores
- ✅ Visual confidence charts and annotated results
- ✅ Comprehensive training pipeline with validation
- ✅ Model evaluation metrics (accuracy, confusion matrix, ROC curve)

## 📁 Project Structure

```
Gallery_Image_Classification/
│
├── data_preprocess.py          # Data loading and preprocessing
├── train.py                    # Model training script
├── app.py                      # Web application (Gradio interface)
├── predict.py                  # Command-line prediction script
├── config.py                   # Configuration settings
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── GalleyImage_Dataset/        # Gallery images (your dataset)
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
│
├── Non_GalleyImage_Dataset/    # Non-Gallery images (your dataset)
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
│
└── models/                     # Saved models (created after training)
    ├── best_model.pth
    ├── final_model.pth
    ├── training_history.png
    ├── confusion_matrix.png
    └── roc_curve.png
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Step 1: Clone or Download the Project

```bash
# Create project directory
mkdir Gallery_Image_Classification
cd Gallery_Image_Classification
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision numpy Pillow matplotlib seaborn scikit-learn gradio tqdm
```

### Step 3: Prepare Your Dataset

1. Create two folders with your images:
   - `GalleyImage_Dataset` - containing Gallery images
   - `Non_GalleyImage_Dataset` - containing Non-Gallery images

2. Supported formats: JPG, JPEG, PNG

3. Recommended: At least 100-500 images per class for good results

## 🎯 Usage

### Step 1: Data Preprocessing

First, run the preprocessing script to verify your dataset:

```bash
python data_preprocess.py
```

This will:
- Load and validate your dataset
- Display dataset statistics
- Create sample visualization
- Test data loaders

**Expected Output:**
```
============================================================
Gallery Image Classification - Data Preprocessing
============================================================

1. Loading dataset...
Found 500 Gallery images
Found 500 Non-Gallery images

Total images loaded: 1000

2. Visualizing sample images...
Sample images saved as 'sample_images.png'

3. Splitting dataset...
Dataset Split:
Training set: 700 images
Validation set: 150 images
Test set: 150 images
...
```

### Step 2: Train the Model

Run the training script:

```bash
python train.py
```

**Training Configuration:**
- Batch Size: 32
- Epochs: 50
- Learning Rate: 0.001
- Optimizer: Adam
- Architecture: ResNet-18 (pretrained on ImageNet)

**Training Process:**
- The model will train for 50 epochs
- Best model is saved automatically based on validation accuracy
- Training history, confusion matrix, and ROC curve are generated
- Checkpoints saved every 10 epochs

**Expected Output:**
```
============================================================
Starting Training...
============================================================

Epoch 1/50
----------------------------------------
Batch [10/22] - Loss: 0.6234
Batch [20/22] - Loss: 0.5123
Train Loss: 0.5876 Acc: 0.7214
Val Loss: 0.4532 Acc: 0.8067
✓ Best model saved! Validation Accuracy: 0.8067
Epoch completed in 2m 15s

...

Training completed!
Best Validation Accuracy: 0.9533
============================================================
```

**Generated Files:**
- `models/best_model.pth` - Best model (highest validation accuracy)
- `models/final_model.pth` - Final model after all epochs
- `models/training_history.png` - Loss and accuracy curves
- `models/confusion_matrix.png` - Confusion matrix on test set
- `models/roc_curve.png` - ROC curve with AUC score

### Step 3: Run the Web Application

Launch the web interface:

```bash
python app.py
```

The application will start at: **http://127.0.0.1:7860**

**Web Interface Features:**
- 📤 Upload images (drag & drop or click)
- 🔍 Real-time analysis with confidence scores
- 📊 Visual confidence charts
- 🖼️ Annotated result images
- 💡 Detailed prediction explanations

## 📊 Model Performance

The model provides several metrics:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

### Confidence Levels
- **Very High** (>95%): Model is very confident
- **High** (85-95%): Model is confident
- **Medium** (70-85%): Model is moderately confident
- **Low** (<70%): Model has lower confidence

## 🎨 Customization

### Modify Training Parameters

Edit `train.py`:

```python
# Hyperparameters
BATCH_SIZE = 32          # Increase for more GPU memory
NUM_EPOCHS = 50          # Train for more epochs
LEARNING_RATE = 0.001    # Adjust learning rate
```

### Modify Data Split

Edit in `data_preprocess.py` or `train.py`:

```python
train_data, val_data, test_data = split_dataset(
    image_paths, labels,
    train_size=0.7,   # 70% for training
    val_size=0.15,    # 15% for validation
    test_size=0.15,   # 15% for testing
    random_state=42
)
```

### Change Model Architecture

Edit `train.py` and `app.py`:

```python
# Use ResNet-34, ResNet-50, or other architectures
self.resnet = models.resnet34(pretrained=pretrained)
# or
self.resnet = models.resnet50(pretrained=pretrained)
```

## 🐛 Troubleshooting

### Issue: "No images found"

**Solution:**
- Check that your dataset paths are correct
- Ensure images are in supported formats (JPG, JPEG, PNG)
- Verify folder names: `GalleyImage_Dataset` and `Non_GalleyImage_Dataset`

### Issue: "CUDA out of memory"

**Solution:**
- Reduce batch size in `train.py`: `BATCH_SIZE = 16` or `BATCH_SIZE = 8`
- Use CPU training by commenting out CUDA checks
- Reduce image size

### Issue: "Model not found" when running app.py

**Solution:**
- Make sure you've trained the model first using `train.py`
- Check that `models/best_model.pth` exists
- If not, run `python train.py` first

### Issue: Low accuracy

**Solution:**
- Increase number of training images (recommended: 500+ per class)
- Train for more epochs
- Increase model complexity (use ResNet-34 or ResNet-50)
- Check data quality and labeling accuracy
- Add more data augmentation

## 📈 Performance Tips

1. **More Data**: Collect more diverse images for better generalization
2. **Data Augmentation**: Already included in preprocessing
3. **Transfer Learning**: Model uses ImageNet pretrained weights
4. **Learning Rate**: Adjust if training is unstable
5. **Regularization**: Dropout and weight decay are applied
6. **Early Stopping**: Monitor validation loss, stop if overfitting

## 🔍 Understanding Results

### Confidence Scores

The model outputs two probabilities that sum to 100%:
- **Gallery Image Probability**: Likelihood of being a Gallery image
- **Non-Gallery Image Probability**: Likelihood of NOT being a Gallery image

### Certainty Levels

- **🟢 Very High (>95%)**: Trust this prediction
- **🟡 High (85-95%)**: Reliable prediction
- **🟠 Medium (70-85%)**: Use with caution
- **🔴 Low (<70%)**: May need manual verification

## 📝 Example Usage

```python
# Example: Using the model programmatically
from app import load_model, predict_image
from PIL import Image

# Load model
load_model('models/best_model.pth')

# Load and predict
image = Image.open('test_image.jpg')
result, chart, annotated, _ = predict_image(image)

print(result)
```

## 🤝 Contributing

Feel free to:
- Add more features
- Improve model architecture
- Enhance UI/UX
- Fix bugs
- Add more documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙋 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure dataset is properly formatted
4. Check console output for error messages

## 🎓 Model Details

**Architecture:** ResNet-18

**Input:** 224x224 RGB images

**Output:** Binary classification (2 classes)

**Training:**
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Learning Rate Schedule: Step LR (decay every 15 epochs)
- Data Augmentation: Random crop, flip, rotation, color jitter
- Normalization: ImageNet statistics

**Inference:**
- Input image resized to 224x224
- Normalized using ImageNet statistics
- Softmax activation for probabilities
- Argmax for final prediction

---

## 🚀 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset in the correct folders
# GalleyImage_Dataset/
# Non_GalleyImage_Dataset/

# 3. Preprocess data (optional, for verification)
python data_preprocess.py

# 4. Train the model
python train.py

# 5. Run the web app
python app.py

# 6. Open browser to http://127.0.0.1:7860
```

**That's it! You're ready to detect Gallery images! 🎉**

---

## 🖥️ Command-Line Predictions

For batch processing or automation, use the command-line prediction tool:

### Single Image Prediction

```bash
python predict.py --image path/to/image.jpg
```

### Batch Prediction (Folder)

```bash
python predict.py --folder path/to/folder
```

### Save Results to File

```bash
# Save as JSON
python predict.py --folder path/to/folder --output results.json --format json

# Save as CSV
python predict.py --folder path/to/folder --output results.csv --format csv
```

### Use Custom Model

```bash
python predict.py --image test.jpg --model models/custom_model.pth
```

**Example Output:**

```
============================================================
PREDICTION RESULT
============================================================

✅ Prediction: Gallery Image
📊 Confidence: 98.45%
🎯 Certainty Level: Very High

Detailed Probabilities:
  Gallery Image: 98.45%
  Non-Gallery Image: 1.55%

📁 Image: test_image.jpg
============================================================
```

---

## ⚙️ Configuration

All settings can be modified in `config.py`:

```python
# Dataset paths
GALLERY_PATH = "path/to/gallery/images"
NON_GALLERY_PATH = "path/to/non/gallery/images"

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Data split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

Print current configuration:

```bash
python config.py
```

---

## 🛠️ Utility Functions

The `utils.py` module provides helpful functions:

```python
from utils import *

# Check for corrupted images
clean_corrupted_images("path/to/folder")

# Get model parameter count
count_parameters(model)

# Calculate class weights for imbalanced data
weights = calculate_class_weights(labels)

# Early stopping during training
early_stopping = EarlyStopping(patience=10)
```

**That's it! You're ready to detect Gallery images! 🎉**