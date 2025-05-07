# Convolutional Neural Network (CNN) Implementation

This directory contains an implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset.

## Dataset

The implementation uses the MNIST dataset, which contains:
- 60,000 training images
- 10,000 test images
- 28x28 pixel grayscale images
- 10 classes (digits 0-9)

## Model Architecture

The CNN model consists of:
1. **First Convolutional Block**:
   - Conv2D layer with 32 filters (3x3)
   - ReLU activation
   - MaxPooling2D (2x2)

2. **Second Convolutional Block**:
   - Conv2D layer with 64 filters (3x3)
   - ReLU activation
   - MaxPooling2D (2x2)

3. **Third Convolutional Block**:
   - Conv2D layer with 64 filters (3x3)
   - ReLU activation

4. **Dense Layers**:
   - Flatten layer
   - Dense layer with 64 units and ReLU activation
   - Dropout layer (0.5)
   - Output layer with 10 units and softmax activation

## Features

- **Data Preprocessing**: Normalizes pixel values and reshapes for CNN input
- **Model Training**: Uses Adam optimizer and categorical crossentropy loss
- **Visualization**: Generates multiple plots to understand the results
- **Evaluation**: Provides accuracy metrics and classification report
- **Model Saving**: Saves the trained model for future use

## Requirements

- numpy
- tensorflow
- matplotlib
- seaborn
- scikit-learn

## Usage

To run the example:

```bash
python cnn_implementation.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Create and compile the CNN model
3. Train the model for 5 epochs
4. Generate visualizations:
   - training_history.png: Shows training and validation accuracy/loss
   - confusion_matrix.png: Shows the confusion matrix
   - sample_predictions.png: Shows example predictions
5. Save the trained model as 'mnist_cnn_model.h5'

## Output Files

- `mnist_cnn_model.h5`: The trained model
- `training_history.png`: Training and validation metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `sample_predictions.png`: Example predictions with images

## Understanding the Results

1. **Training History**: Shows how the model's accuracy and loss change during training
2. **Confusion Matrix**: Shows how well the model performs for each digit
3. **Sample Predictions**: Shows actual images with their true and predicted labels
4. **Classification Report**: Provides precision, recall, and F1-score for each digit 