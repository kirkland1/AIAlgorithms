import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Salary_dataset.csv')

# Prepare X (features) and y (target)
X = df['YearsExperience'].values.reshape(-1, 1)
y = df['Salary'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model results
print(f'Intercept: {model.intercept_:.2f}')
print(f'Coefficient: {model.coef_[0]:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.4f}')

# Example prediction
years_exp = np.array([[5.0]])
predicted_salary = model.predict(years_exp)
print(f'\nPredicted salary for 5 years of experience: ${predicted_salary[0]:.2f}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.legend()
plt.grid(True)
plt.show()

