# Decision Tree Implementation

This directory contains an implementation of a Decision Tree algorithm for predicting student performance based on various factors.

## Dataset

The implementation uses a synthetic dataset with the following features:
- Study Hours: Number of hours spent studying per day
- Attendance: Percentage of classes attended
- Previous Score: Score in the previous exam
- Sleep Hours: Number of hours of sleep per day

The target variable is a binary classification:
- 0: Fail
- 1: Pass

## Implementation Details

The decision tree implementation includes:
1. Dataset generation with realistic student performance patterns
2. Decision tree training with scikit-learn
3. Model evaluation with accuracy and classification report
4. Visualization of the decision tree
5. Example predictions with actual vs predicted results

## Features

- **Data Generation**: Creates a synthetic dataset with 100 samples
- **Model Training**: Uses scikit-learn's DecisionTreeClassifier
- **Visualization**: Generates a visual representation of the decision tree
- **Evaluation**: Provides accuracy metrics and classification report
- **Predictions**: Shows example predictions with actual results

## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib

## Usage

To run the example:

```bash
python decision_tree_implementation.py
```

This will:
1. Generate a synthetic student performance dataset
2. Save the dataset to 'student_dataset.csv'
3. Train a decision tree model
4. Display model accuracy and classification report
5. Generate a visualization of the decision tree (saved as 'decision_tree.png')
6. Show example predictions with actual results

## Output Files

- `student_dataset.csv`: The generated dataset
- `decision_tree.png`: Visualization of the trained decision tree

## Parameters

The decision tree is configured with:
- max_depth=4: Limits the depth of the tree to prevent overfitting
- random_state=42: For reproducibility 