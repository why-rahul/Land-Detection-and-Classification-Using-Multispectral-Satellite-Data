# Land Detection and Classification Using Multispectral Satellite Data

## Overview

This project performs land classification using multispectral satellite images from the EuroSAT dataset. Deep learning models are used to classify satellite images into different land cover categories.

## Dataset

- Dataset: EuroSAT Multispectral
- Number of Classes: 10
- Number of Bands: 13
- Image Size: 64 × 64 pixels

### Classes
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

## Models Used

### Custom CNN
- Designed for 13-channel multispectral input
- Achieved 95.23% test accuracy

### Modified ResNet18
- First convolution layer modified from 3 channels to 13 channels
- Achieved 82.56% test accuracy

## Data Preprocessing

- TIFF image loading
- Tensor conversion
- Min-Max normalization
- Image resizing to 64 × 64
- Train/Validation/Test split

## Results

| Model | Accuracy |
|--------|----------|
| CNN | 95.23% |
| ResNet18 | 82.56% |

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- TIFF Processing

## Author

Rahul Jana

M.Sc. Computer Science  
Ramakrishna Mission Vivekananda Educational and Research Institute
