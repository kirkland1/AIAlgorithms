# Principal Component Analysis (PCA) Implementation

This directory contains an implementation of Principal Component Analysis (PCA) using a customer behavior dataset.

## Dataset

The implementation uses a synthetic dataset with the following features:
- Age: Customer's age
- Income: Annual income in dollars
- Spending: Monthly spending in dollars
- Visits: Number of store visits per month
- Loyalty Score: Customer loyalty score (0-100)

The dataset includes 100 customer records with correlated features to demonstrate PCA's ability to identify patterns and reduce dimensionality.

## Implementation Details

The PCA implementation includes:
1. Dataset generation with realistic customer behavior patterns
2. Feature standardization using StandardScaler
3. PCA transformation using scikit-learn
4. Visualization of results
5. Analysis of feature importance in principal components

## Features

- **Data Generation**: Creates a synthetic dataset with 100 samples
- **PCA Analysis**: Reduces 5 features to 2 principal components
- **Visualization**: Generates multiple plots to understand the results
- **Feature Importance**: Shows how each original feature contributes to the principal components

## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage

To run the example:

```bash
python pca_implementation.py
```

This will:
1. Generate a synthetic customer behavior dataset
2. Save the dataset to 'customer_dataset.csv'
3. Perform PCA analysis
4. Display explained variance for each principal component
5. Generate visualizations:
   - variance_explained.png: Shows how much variance each PC explains
   - pca_scatter.png: Scatter plot of the first two PCs
   - feature_importance.png: Shows how original features contribute to PCs

## Output Files

- `customer_dataset.csv`: The generated dataset
- `variance_explained.png`: Visualization of explained variance
- `pca_scatter.png`: Scatter plot of principal components
- `feature_importance.png`: Feature importance visualization

## Understanding the Results

1. **Explained Variance**: Shows how much of the total variance is captured by each principal component
2. **PCA Scatter Plot**: Shows the data projected onto the first two principal components
3. **Feature Importance**: Shows how each original feature contributes to the principal components 