import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def create_student_dataset(n_samples=100):
    """Create a synthetic dataset for student performance prediction"""
    np.random.seed(42)  # For reproducibility
    
    # Generate features
    study_hours = np.random.normal(5, 1.5, n_samples)  # Mean 5 hours, std 1.5
    attendance = np.random.normal(85, 10, n_samples)   # Mean 85%, std 10%
    previous_score = np.random.normal(75, 15, n_samples)  # Mean 75%, std 15%
    sleep_hours = np.random.normal(7, 1, n_samples)    # Mean 7 hours, std 1
    
    # Create a DataFrame
    df = pd.DataFrame({
        'study_hours': study_hours,
        'attendance': attendance,
        'previous_score': previous_score,
        'sleep_hours': sleep_hours
    })
    
    # Generate target variable (Pass/Fail) based on rules
    # Students are more likely to pass if they:
    # - Study more than 4 hours
    # - Have attendance above 80%
    # - Have previous score above 70%
    # - Sleep between 6-8 hours
    pass_probability = (
        (df['study_hours'] > 4).astype(int) * 0.3 +
        (df['attendance'] > 80).astype(int) * 0.3 +
        (df['previous_score'] > 70).astype(int) * 0.3 +
        ((df['sleep_hours'] >= 6) & (df['sleep_hours'] <= 8)).astype(int) * 0.1
    )
    
    # Add some randomness
    pass_probability += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary outcome
    df['result'] = (pass_probability > 0.5).astype(int)
    
    # Save to CSV
    df.to_csv('student_dataset.csv', index=False)
    
    return df

def train_decision_tree(X, y):
    """Train a decision tree classifier"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print feature importance
    print("\nFeature Importance:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    print(feature_importance)
    
    # Explain root node selection
    root_feature = feature_importance.iloc[0]['Feature']
    root_importance = feature_importance.iloc[0]['Importance']
    print(f"\nRoot Node Selection:")
    print(f"The root node is '{root_feature}' with importance {root_importance:.3f}")
    print("This feature was chosen because it provides the best split for separating Pass/Fail classes")
    print("The decision tree algorithm uses Gini impurity to determine the best split at each node")
    
    return clf, X_train, X_test, y_train, y_test

def plot_decision_tree(clf, feature_names):
    """Plot the decision tree"""
    plt.figure(figsize=(20,10))
    plot_tree(clf, feature_names=list(feature_names), class_names=['Fail', 'Pass'],
              filled=True, rounded=True, fontsize=10)
    plt.savefig('decision_tree.png')
    plt.show()
    plt.close()

def main():
    # Create the dataset
    print("Creating student performance dataset...")
    df = create_student_dataset(100)
    
    # Prepare features and target
    X = df[['study_hours', 'attendance', 'previous_score', 'sleep_hours']]
    y = df['result']
    
    # Train the model
    print("\nTraining decision tree model...")
    clf, X_train, X_test, y_train, y_test = train_decision_tree(X, y)
    
    # Plot the decision tree
    print("\nPlotting decision tree...")
    plot_decision_tree(clf, X.columns)
    
    # Print some example predictions
    print("\nExample Predictions:")
    print("Study Hours | Attendance | Previous Score | Sleep Hours | Predicted Result")
    print("-" * 75)
    
    # Get 5 random test examples
    random_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for idx in random_indices:
        features = X_test.iloc[idx]
        prediction = "Pass" if clf.predict([features])[0] == 1 else "Fail"
        actual = "Pass" if y_test.iloc[idx] == 1 else "Fail"
        print(f"{features['study_hours']:11.1f} | {features['attendance']:10.1f} | "
              f"{features['previous_score']:14.1f} | {features['sleep_hours']:11.1f} | "
              f"{prediction:14s} (Actual: {actual})")

if __name__ == "__main__":
    main() 