import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        """Store the training data"""
        self.X_train = X
        self.y_train = y
        
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def get_neighbors(self, x):
        """Find k nearest neighbors of x"""
        distances = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        
        # Sort distances and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def predict_classification(self, x):
        """Predict class for a single sample"""
        neighbors = self.get_neighbors(x)
        # Get the most common class among neighbors
        classes = [neighbor[1] for neighbor in neighbors]
        return Counter(classes).most_common(1)[0][0]
    
    def predict(self, X):
        """Predict for multiple samples"""
        return np.array([self.predict_classification(x) for x in X])

def plot_decision_boundary(X, y, model, title):
    """Plot decision boundary for 2D data"""
    # Create a mesh grid with fewer points
    h = 0.5  # Increased step size to reduce number of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions for the mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Process mesh points in batches to avoid memory issues
    batch_size = 1000
    Z = np.zeros(len(mesh_points))
    
    for i in range(0, len(mesh_points), batch_size):
        batch = mesh_points[i:i + batch_size]
        Z[i:i + batch_size] = model.predict(batch)
    
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='RdYlBu')
    
    # Add labels and title
    plt.title(title)
    plt.xlabel('Weight (grams)')
    plt.ylabel('Texture (1-10)')
    
    # Add a colorbar
    plt.colorbar(label='Fruit Type (0=Apple, 1=Orange)')
    
    # Show the plot
    plt.show()

def create_fruit_dataset():
    """Create a dataset of apples and oranges based on weight and texture"""
    # Apple data (class 0)
    apple_weights = np.random.normal(150, 20, 50)  # Mean weight 150g
    apple_textures = np.random.normal(7, 1, 50)    # Mean texture 7
    
    # Orange data (class 1)
    orange_weights = np.random.normal(200, 25, 50)  # Mean weight 200g
    orange_textures = np.random.normal(4, 1, 50)    # Mean texture 4
    
    # Combine the data
    X = np.vstack([
        np.column_stack((apple_weights, apple_textures)),
        np.column_stack((orange_weights, orange_textures))
    ])
    
    # Create labels (0 for apples, 1 for oranges)
    y = np.array([0] * 50 + [1] * 50)
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame({
        'weight': X[:, 0],
        'texture': X[:, 1],
        'fruit': ['Apple' if label == 0 else 'Orange' for label in y]
    })
    df.to_csv('fruit_dataset.csv', index=False)
    
    return X, y

def main():
    # Create the fruit dataset
    X, y = create_fruit_dataset()
    
    # Split the data (80% training, 20% testing)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train the model
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Classification Accuracy: {accuracy:.2f}")
    
    # Plot decision boundary
    plot_decision_boundary(X_train, y_train, knn, "KNN Decision Boundary (Apples vs Oranges)")
    
    # Print random example predictions
    print("\nExample Predictions:")
    print("Weight (g) | Texture | Predicted Fruit | Actual Fruit")
    print("-" * 50)
    
    # Get 5 random indices from test set
    random_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for idx in random_indices:
        weight, texture = X_test[idx]
        prediction = "Apple" if y_pred[idx] == 0 else "Orange"
        actual = "Apple" if y_test[idx] == 0 else "Orange"
        print(f"{weight:9.1f} | {texture:7.1f} | {prediction:14s} | {actual}")

if __name__ == "__main__":
    main() 