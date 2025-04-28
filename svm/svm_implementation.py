import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def create_email_dataset(n_samples=100):
    """Create a synthetic dataset for email classification"""
    np.random.seed(42)  # For reproducibility
    
    # Generate features
    word_count = np.random.normal(200, 50, n_samples)  # Mean 200 words, std 50
    link_count = np.random.normal(2, 1, n_samples)     # Mean 2 links, std 1
    capital_ratio = np.random.normal(0.2, 0.1, n_samples)  # Mean 20% capitals, std 10%
    exclamation_count = np.random.normal(3, 2, n_samples)  # Mean 3 exclamations, std 2
    urgency_words = np.random.normal(1, 1, n_samples)  # Mean 1 urgency word, std 1
    
    # Create a DataFrame
    df = pd.DataFrame({
        'word_count': word_count,
        'link_count': link_count,
        'capital_ratio': capital_ratio,
        'exclamation_count': exclamation_count,
        'urgency_words': urgency_words
    })
    
    # Generate target variable (Spam/Not Spam) based on rules
    # Emails are more likely to be spam if they:
    # - Have many links
    # - Have high capital ratio
    # - Have many exclamation marks
    # - Have urgency words
    spam_probability = (
        (df['link_count'] > 2).astype(int) * 0.3 +
        (df['capital_ratio'] > 0.3).astype(int) * 0.3 +
        (df['exclamation_count'] > 3).astype(int) * 0.2 +
        (df['urgency_words'] > 1).astype(int) * 0.2
    )
    
    # Add some randomness
    spam_probability += np.random.normal(0, 0.1, n_samples)
    
    # Convert to binary outcome
    df['is_spam'] = (spam_probability > 0.5).astype(int)
    
    # Save to CSV
    df.to_csv('email_dataset.csv', index=False)
    
    return df

def train_svm(X, y):
    """Train an SVM classifier"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model with adjusted parameters
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return svm, scaler, X_train, X_test, y_train, y_test

def plot_feature_importance(X, y):
    """Plot feature distributions for spam vs not spam"""
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(X.columns):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='is_spam', y=feature, data=pd.concat([X, pd.Series(y, name='is_spam')], axis=1))
        plt.title(f'{feature} Distribution')
        plt.xlabel('Is Spam')
        plt.ylabel(feature)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def main():
    # Create the dataset
    print("Creating email classification dataset...")
    df = create_email_dataset(100)
    
    # Prepare features and target
    X = df[['word_count', 'link_count', 'capital_ratio', 'exclamation_count', 'urgency_words']]
    y = df['is_spam']
    
    # Plot feature distributions
    print("\nPlotting feature distributions...")
    plot_feature_importance(X, y)
    
    # Train the model
    print("\nTraining SVM model...")
    svm, scaler, X_train, X_test, y_train, y_test = train_svm(X, y)
    
    # Print some example predictions
    print("\nExample Predictions:")
    print("Word Count | Link Count | Capital Ratio | Exclamation Count | Urgency Words | Predicted | Actual")
    print("-" * 100)
    
    # Get 5 random test examples
    random_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for idx in random_indices:
        features = X_test.iloc[idx]
        # Convert features to DataFrame to preserve feature names
        features_df = pd.DataFrame([features], columns=X.columns)
        # Scale the features before prediction
        features_scaled = scaler.transform(features_df)
        prediction = "Spam" if svm.predict(features_scaled)[0] == 1 else "Not Spam"
        actual = "Spam" if y_test.iloc[idx] == 1 else "Not Spam"
        print(f"{features['word_count']:10.1f} | {features['link_count']:10.1f} | "
              f"{features['capital_ratio']:12.2f} | {features['exclamation_count']:16.1f} | "
              f"{features['urgency_words']:12.1f} | {prediction:9s} | {actual}")

if __name__ == "__main__":
    main() 