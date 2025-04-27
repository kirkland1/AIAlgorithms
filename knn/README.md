# K-Nearest Neighbors (KNN) Implementation

This directory contains an implementation of the K-Nearest Neighbors algorithm from scratch, demonstrated using a simple fruit classification example (apples vs oranges).

## Implementation Details

The KNN algorithm is implemented in `knn_implementation.py` and includes:

1. A `KNN` class for classification
2. Euclidean distance calculation
3. K-nearest neighbors selection
4. Prediction method for classification
5. Visualization of decision boundaries

## Features

- **Classification**: Uses majority voting among k-nearest neighbors
- **Visualization**: Includes a function to plot decision boundaries
- **Example**: Uses a custom dataset of apples and oranges based on:
  - Weight (in grams)
  - Texture (scale of 1-10)

## Requirements

- numpy
- matplotlib

## Usage

To run the example:

```bash
python knn_implementation.py
```

This will:
1. Generate a synthetic dataset of apples and oranges
2. Split it into training and test sets
3. Train a KNN classifier
4. Make predictions
5. Display the accuracy
6. Show a visualization of the decision boundary
7. Print example predictions

## Custom Usage

To use the KNN implementation with your own data:

```python
from knn_implementation import KNN

# Create and train the model
knn = KNN(k=3)  # k is the number of neighbors
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
```

## Parameters

- `k`: Number of nearest neighbors to consider (default=3) 