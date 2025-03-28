# Fraud Detection using ReNN Model

This repository contains a machine learning project for fraud detection using a **Recurrent Neural Network (ReNN)** model. The model is trained using a dataset of financial transactions, and it includes preprocessing steps, handling of class imbalance, feature scaling, and model evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Project Overview

The objective of this project is to build and train a neural network model for detecting fraudulent financial transactions. The dataset used in this project contains a mixture of legitimate and fraudulent transaction records, with an imbalanced distribution of classes. The project handles the imbalance through **SMOTE** (Synthetic Minority Over-sampling Technique) and applies **feature scaling** for model training.

The custom neural network architecture is called **ReNN**, which combines traditional neural network layers with external rules based on certain features (like the transaction amount).

## Data Preprocessing

1. **Loading the Data**: The dataset is loaded from a CSV file (`fraudTest.csv`).
2. **Handling Duplicates**: Any duplicate records are removed.
3. **Feature Scaling**: Features are standardized using `StandardScaler` to ensure that they have zero mean and unit variance.
4. **Class Imbalance**: The class imbalance is handled using **SMOTE**, which generates synthetic samples for the minority class.
5. **Train-Test Split**: The data is split into a training set (80%) and a testing set (20%).

## Model Architecture

The model used for fraud detection is a custom neural network architecture called **ReNN (Recurrent Neural Network)**. This model is designed to take both the original features (e.g., transaction details) and additional features derived from external rules (e.g., rules based on transaction amounts). 

### ReNN Architecture
- **Input Layer**: Takes the features of the transactions.
- **Rule Layer**: Takes additional rule-based information (like whether the amount exceeds a certain threshold).
- **Hidden Layers**: Consists of one or more hidden layers with ReLU activations.
- **Output Layer**: The output is a single value between 0 and 1, passed through a sigmoid function to represent the probability of fraud.

## Training the Model

1. **Epochs**: The model is trained for 300 epochs with the Adam optimizer and binary cross-entropy loss.
2. **Class Weighting**: Class imbalance is addressed by assigning higher weights to the minority class.
3. **Validation**: During training, the model's loss is monitored, and after training, the model is evaluated on a test set.

## Model Evaluation

After training, the model's performance is evaluated using the following metrics:
- **Accuracy**: The overall accuracy of the model on the test set.
- **Classification Report**: Precision, recall, and F1-score for each class (fraud and non-fraud).
- **Confusion Matrix**: Displays the confusion matrix for further performance analysis.

### Evaluation Code Example

```python
accuracy = accuracy_score(y_test_np, predictions_np)
print(f'Accuracy: {accuracy:.4f}')

report = classification_report(y_test_np, predictions_np, target_names=['Class 0', 'Class 1'])
print(report)
