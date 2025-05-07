import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def create_cnn_model():
    """Create a CNN model for MNIST digit classification"""
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    plt.close()

def plot_sample_predictions(X_test, y_test, y_pred, num_samples=5):
    """Plot sample predictions with their true and predicted labels"""
    plt.figure(figsize=(15, 3))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_test[i]}\nPred: {y_pred[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()
    plt.close()

def plot_feature_maps(model, X_test, y_test, layer_index=0, num_samples=5):
    """Plot feature maps from the first convolutional layer"""
    # Create a model that outputs the feature maps
    feature_map_model = models.Model(inputs=model.input, 
                                   outputs=model.layers[layer_index].output)
    
    # Get feature maps for the first few test images
    feature_maps = feature_map_model.predict(X_test[:num_samples])
    
    # Plot the feature maps
    plt.figure(figsize=(15, 3*num_samples))
    
    for i in range(num_samples):
        # Plot original image
        plt.subplot(num_samples, 9, i*9 + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Original\nDigit: {y_test[i]}')
        plt.axis('off')
        
        # Plot first 8 feature maps
        for j in range(8):
            plt.subplot(num_samples, 9, i*9 + j + 2)
            plt.imshow(feature_maps[i, :, :, j], cmap='viridis')
            plt.title(f'Feature {j+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_maps.png')
    plt.show()
    plt.close()

def plot_error_analysis(X_test, y_test, y_pred):
    """Plot examples of misclassified digits"""
    # Find misclassified examples
    misclassified = np.where(y_test != y_pred)[0]
    
    if len(misclassified) > 0:
        plt.figure(figsize=(15, 3))
        for i in range(min(5, len(misclassified))):
            idx = misclassified[i]
            plt.subplot(1, 5, i+1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('misclassified_examples.png')
        plt.show()
        plt.close()

def main():
    # Load and preprocess MNIST dataset
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    
    # Reshape for CNN input
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Create and compile the model
    print("\nCreating CNN model...")
    model = create_cnn_model()
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Print model summary
    model.summary()
    
    # Train the model
    print("\nTraining the model...")
    history = model.fit(X_train, y_train,
                       epochs=5,
                       batch_size=64,
                       validation_split=0.2)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred_classes)
    
    # Plot sample predictions
    print("Plotting sample predictions...")
    plot_sample_predictions(X_test, y_test, y_pred_classes)
    
    # Plot feature maps
    print("Plotting feature maps...")
    plot_feature_maps(model, X_test, y_test)
    
    # Plot error analysis
    print("Plotting error analysis...")
    plot_error_analysis(X_test, y_test, y_pred_classes)
    
    # Save the model in the new format
    model.save('mnist_cnn_model.keras')
    print("\nModel saved as 'mnist_cnn_model.keras'")

if __name__ == "__main__":
    main() 