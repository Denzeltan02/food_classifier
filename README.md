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
├── .ipynb_checkpoints/                             # Jupyter notebook checkpoint folder
├── artifacts/                                      # Saved model weights, logs, configs, etc.
├── dataset_split/                                  # Dataset partitions for training, validation, and testing
├── food_efficientnet_b3.tf                         # TF version of EfficientNet-B3 model
├── food_efficientnet_b3_grayscale_robust_tf        # TF version of grayscale-robust EfficientNet-B3
├── smaller_dataset/                                # Reduced-size dataset for quicker experiments
├── .gitattributes                                  # Git attributes for repository configuration
├── app.py                                          # Hugging Face web application
├── class_names.json                                # JSON file mapping class indices to food labels
├── efficientnet_backbone_pruned.png                # Visualization of the pruned EfficientNet backbone
├── food_classifier.ipynb                           # Primary Jupyter notebook for training/evaluation of food classification
├── food_detection.ipynb                            # Primary Jupyter notebook for training/evaluation of food detection
├── food_efficientnet_b0.keras                      # Trained EfficientNet-B0 model
├── food_efficientnet_b0_grayscale_robust.keras     # Grayscale-robust EfficientNet-B0 model
├── food_efficientnet_b3.keras                      # Trained EfficientNet-B3 model
├── food_efficientnet_b3_grayscale_robust.keras     # Grayscale-robust EfficientNet-B3 model
├── food_efficientnet_b3_tuned.keras                # Tuned version of EfficientNet-B3 model
└── requirements.txt                                # Python dependencies
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







