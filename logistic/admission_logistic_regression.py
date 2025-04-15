import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('student_admission2.csv')

# Prepare features (X) and target (y)
X = df[['exam1_score', 'exam2_score']].values
y = df['admitted'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model coefficients and intercept
print("Logistic Regression Equation:")
print(f"log(p/(1-p)) = {model.intercept_[0]:.2f} + {model.coef_[0][0]:.2f}×exam1_score + {model.coef_[0][1]:.2f}×exam2_score")

# Print model performance
print("\nModel Performance:")
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# Add detailed confusion matrix analysis
print("\nConfusion Matrix Analysis:")
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Calculate additional metrics
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)

print("\nDetailed Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Create a visualization
plt.figure(figsize=(10, 8))

# Plot training data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Not Admitted', c='red', marker='x')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Admitted', c='green', marker='o')

# Create a mesh grid to plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Make predictions on mesh grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# Example predictions
print("\nExample Predictions:")
example_students = np.array([
    [70, 85],  # High scores
    [45, 50],  # Low scores
    [60, 75]   # Moderate scores
])

predictions = model.predict(example_students)
probabilities = model.predict_proba(example_students)

for i, (scores, pred, prob) in enumerate(zip(example_students, predictions, probabilities)):
    print(f"\nStudent {i+1}:")
    print(f"Exam 1 Score: {scores[0]}, Exam 2 Score: {scores[1]}")
    print(f"Prediction: {'Admitted' if pred == 1 else 'Not Admitted'}")
    print(f"Probability of Admission: {prob[1]:.2%}")

# Plot decision boundary
plt.contour(xx, yy, Z, colors='black', levels=[0.5])
plt.contourf(xx, yy, Z, alpha=0.1, levels=[0, 0.5, 1])

plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.title('Student Admission Prediction')
plt.legend()
plt.grid(True)
plt.savefig('admission_decision_boundary.png')
plt.show()