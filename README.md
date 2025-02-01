# Malware Detection using Machine Learning and Quantum Techniques

## Overview
This project aims to detect malware using various machine learning models, including traditional classifiers, deep learning approaches, and quantum-enhanced techniques. The dataset, obtained from a hackathon, underwent extensive preprocessing, feature engineering, and dimensionality reduction to optimize performance and computational efficiency.

## Dataset Details
- **Training Data**: 22,017 samples, 28,696 features
- **Test Data**: 1,480 samples, 22,690 features
- **Features**:
  - SHA256 (Unique Identifier)
  - Type (Malware/Benign Classification)
  - API Functions
  - DLL Imports
  - Portable Executable Features

## Preprocessing and Feature Engineering
1. **Data Merging**: Combined multiple datasets into a unified format.
2. **Dimensionality Reduction**: Applied **Principal Component Analysis (PCA)**
   - 100 principal components for traditional ML and ConvLSTM models.
   - 50 principal components for Quantum XGBoost model.
3. **Standardization**: Ensured uniform feature scaling.
4. **Class Balance Handling**: Addressed class imbalances in the dataset.

## Models Trained
The following models were trained to evaluate performance:

- **Traditional Machine Learning Models**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - Gradient Boosting Classifier
  
- **Deep Learning Models**
  - Multi-Layer Perceptron (MLP) Classifier
  - ConvLSTM (Hybrid Convolutional + LSTM Model)
  
- **Quantum Machine Learning**
  - Quantum XGBoost

## Best Performing Model
The **MLP Classifier** was selected as the best-performing model due to:
- High accuracy and F1-score
- Ability to handle nonlinear relationships
- Robust generalization to unseen data

### Architecture of MLP Classifier
- **Hidden Layers**: 50 neurons (first layer), 25 neurons (second layer)
- **Activation Function**: ReLU
- **Regularization**: L2 (alpha = 0.05)
- **Optimizer**: Adam
- **Training Iterations**: 400 epochs

## Model Deployment Pipeline
A pipeline was developed to:
- Convert incoming data into PCA-transformed format.
- Pass the transformed data into the trained MLP classifier for malware detection.

## Usage Instructions
1. Clone the repository and install dependencies:
   ```sh
   git clone <repository-url>
   cd malware-detection
   pip install -r requirements.txt
   ```
2. Open the Jupyter Notebook:
   ```sh
   jupyter notebook Malware_Detection_Neural_Nomads.ipynb
   ```
3. Run each cell sequentially to preprocess the dataset, train models, and evaluate performance.
4. Modify and test different models as needed.

## Future Work
- Experiment with transformer-based deep learning models.
- Explore hybrid quantum-classical approaches.
- Optimize feature extraction techniques for improved detection rates.

## Authors
- **Neural Nomads Team**
