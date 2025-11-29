# Food Classifier

A computer vision model for detecting and classifying food images using EfficientNet and YOLOv8. The main models used in this project are the EfficientNet-B3 TensorFlow models, with baseline B0 models included.

## Features

### Food Classfication
- Built on **EfficientNet-B3** backbone
- Supports both color and grayscale inputs
- Interactive Hugging Face Space for quick model testing in your browser
- Includes confusion matrices, accuracy plots, and metrics.
- Can be fine-tuned on other food datasets

### Food Detection
- Fine tuned YOLOv8 detection model for identifying multiple food items in a single image
- Supports bounding box visualization
- Trained on annotated food datasets

## Project Structure
```
food_classifier/
├── app.py                          # Hugging Face / Gradio web application
├── calibration_model.keras         # Calibration model for confidence adjustment
├── class_names.json                # JSON file mapping class indices to food labels
├── requirements.txt                # Python dependencies
│
├── food_classifier.ipynb           # Notebook for training/evaluation of food classification
├── food_detection.ipynb            # Notebook for training/evaluation of food detection
├── data_processing.ipynb           # Data preprocessing and augmentation notebook
├── detection_comparison.ipynb      # Comparison of detection approaches
│
├── models_tf/                      # TensorFlow SavedModel exports
│   ├── food_efficientnet_b3/       # EfficientNet-B3 (color) SavedModel
│   └── food_efficientnet_b3_grayft/# EfficientNet-B3 (grayscale fine-tuned) SavedModel
│
├── runs/                           # YOLO training runs and weights
│   └── detect/                     # Detection model checkpoints
|       ├── compare_yolo11n_dishdet/# YOLO11-n comparison experiment
|       |   ├── weights/            # best.pt
|       |   ├── results.csv         # metrics across epochs
│       |   └── results.png         # training curves
│       |
|       ├── compare_yolov8s_dishdet/# YOLOv8-s comparison experiment
|       └── train_dishdet_multi3/   # Final chosen YOLOv8n dish detection model
|
├── data_clean/                     # Cleaned classification dataset
│   ├── train/                      # Training images (20 food classes)
│   ├── val/                        # Validation images
│   ├── test/                       # Test images
│   ├── prediction/                 # Sample images for inference demos
│   └── manifest.csv                # Dataset manifest file
│
├── data_yolo_dish/                 # YOLO-format detection dataset
│   ├── train/                      # Training images + labels
│   ├── valid/                      # Validation images + labels
│   ├── test/                       # Test images + labels
│   └── data.yaml                   # YOLO dataset configuration
│
└── smaller_dataset/                # Reduced dataset for quick experiments
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/food_classifier.git
cd food_classifier
pip install -r requirements.txt
```

## Usage

After installation run app.py to run it locally

```bash
python app.py
```

Or

You can test the model directly in your browser here:
[Hugging Face Space – Food Classifier](https://huggingface.co/spaces/3001sit/FoodClassiferTest)







