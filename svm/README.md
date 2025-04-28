# Support Vector Machine (SVM) Implementation

This directory contains an implementation of Support Vector Machine (SVM) for email classification (spam vs. not spam).

## Dataset

The implementation uses a synthetic dataset with the following features:
- Word Count: Number of words in the email
- Link Count: Number of links in the email
- Capital Ratio: Ratio of capital letters to total letters
- Exclamation Count: Number of exclamation marks
- Urgency Words: Number of urgency-related words

The target variable is a binary classification:
- 0: Not Spam
- 1: Spam

## Implementation Details

The SVM implementation includes:
1. Dataset generation with realistic email patterns
2. Feature visualization using box plots
3. SVM training with RBF kernel
4. Model evaluation with accuracy and classification report
5. Confusion matrix visualization
6. Example predictions with actual results

## Features

- **Data Generation**: Creates a synthetic dataset with 100 samples
- **Model Training**: Uses scikit-learn's SVC with RBF kernel
- **Visualization**: Generates multiple plots to understand the data and results
- **Evaluation**: Provides accuracy metrics and classification report
- **Predictions**: Shows example predictions with actual results

## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Usage

To run the example:

```bash
python svm_implementation.py
```

This will:
1. Generate a synthetic email classification dataset
2. Save the dataset to 'email_dataset.csv'
3. Create feature distribution plots
4. Train an SVM model
5. Display model accuracy and classification report
6. Generate visualizations:
   - feature_distributions.png: Shows feature distributions for spam vs not spam
   - confusion_matrix.png: Shows the confusion matrix
7. Show example predictions with actual results

## Output Files

- `email_dataset.csv`: The generated dataset
- `feature_distributions.png`: Visualization of feature distributions
- `confusion_matrix.png`: Visualization of the confusion matrix

## Understanding the Results

1. **Feature Distributions**: Shows how each feature differs between spam and not spam emails
2. **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives
3. **Classification Report**: Provides precision, recall, and F1-score for each class
4. **Example Predictions**: Shows how the model performs on specific examples 