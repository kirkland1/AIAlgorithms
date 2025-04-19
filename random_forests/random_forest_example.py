import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

def generate_weather_data(n_samples=1000):
    """Generate synthetic weather data for rain prediction"""
    np.random.seed(42)
    
    # Generate features with realistic ranges
    temperature = np.random.normal(25, 5, n_samples)  # Celsius
    humidity = np.random.normal(65, 15, n_samples)    # Percentage
    wind_speed = np.random.normal(15, 5, n_samples)   # km/h
    cloud_cover = np.random.normal(50, 20, n_samples) # Percentage
    pressure = np.random.normal(1013, 10, n_samples)  # hPa
    
    # Create realistic rain conditions based on weather features
    rain = np.zeros(n_samples)
    for i in range(n_samples):
        # Higher chance of rain with:
        # - High humidity (>70%)
        # - High cloud cover (>60%)
        # - Low pressure (<1010 hPa)
        # - Moderate wind speed (10-20 km/h)
        rain_prob = 0
        if humidity[i] > 70:
            rain_prob += 0.3
        if cloud_cover[i] > 60:
            rain_prob += 0.3
        if pressure[i] < 1010:
            rain_prob += 0.2
        if 10 <= wind_speed[i] <= 20:
            rain_prob += 0.2
        
        # Temperature effect (less likely to rain in very hot or cold conditions)
        if temperature[i] > 30 or temperature[i] < 10:
            rain_prob *= 0.5
        
        # Add some randomness
        rain_prob += np.random.normal(0, 0.1)
        
        # Convert probability to binary outcome
        rain[i] = 1 if rain_prob > 0.5 else 0
    
    # Create DataFrame
    data = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,
        'pressure': pressure,
        'rain': rain
    })
    
    return data

def display_dataset_info(data):
    """Display information about the dataset"""
    print("\nDataset Information:")
    print(f"Number of samples: {len(data)}")
    print(f"Number of features: {len(data.columns) - 1}")  # Excluding target
    print(f"Rain distribution:\n{data['rain'].value_counts()}")
    
    print("\nFirst 10 rows of the dataset:")
    print(data.head(10))
    
    print("\nDataset Statistics:")
    print(data.describe())
    
    # Save to CSV
    data.to_csv('weather_dataset.csv', index=False)
    print("\nDataset saved to 'weather_dataset.csv'")

def plot_feature_distributions(data):
    """Plot distributions of features by rain status"""
    plt.figure(figsize=(15, 12))
    
    features = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'pressure']
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i+1)
        sns.histplot(data=data, x=feature, hue='rain', kde=True)
        plt.title(f'Distribution of {feature.replace("_", " ").title()}')
    
    plt.tight_layout()
    plt.savefig('weather_distributions.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance.sort_values().plot(kind='barh')
    plt.title('Feature Importance for Rain Prediction')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix with detailed annotations"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Rain', 'Rain'],
                yticklabels=['No Rain', 'Rain'])
    
    # Add labels and title
    plt.title('Confusion Matrix for Rain Prediction\n(Actual vs Predicted)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics_text = (f'Accuracy: {accuracy:.2f}\n'
                   f'Precision: {precision:.2f}\n'
                   f'Recall: {recall:.2f}\n'
                   f'F1-Score: {f1:.2f}')
    
    plt.text(2.5, 0.5, metrics_text, 
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Generate weather data
    print("Generating weather data...")
    data = generate_weather_data(n_samples=1000)
    
    # Display dataset information and save to CSV
    display_dataset_info(data)
    
    # Split data into features and target
    X = data[['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'pressure']].values
    y = data['rain'].values
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot enhanced confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_feature_distributions(data)
    plot_feature_importance(rf, ['Temperature', 'Humidity', 'Wind Speed', 'Cloud Cover', 'Pressure'])
    
    print("\nVisualizations saved as:")
    print("- weather_distributions.png")
    print("- feature_importance.png")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    main() 