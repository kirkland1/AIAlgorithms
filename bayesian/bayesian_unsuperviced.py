import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=100):
    # Generate correlated features
    
    # Base features with some natural correlations
    study_hours = np.random.normal(6, 2, n_samples)  # Mean of 6 hours, std of 2
    
    # Sleep hours correlates somewhat with study hours (tired students study less)
    sleep_hours = 8 - 0.3 * study_hours + np.random.normal(0, 1, n_samples)
    
    # Stress level increases with study hours but decreases with sleep
    stress_level = (0.3 * study_hours - 0.4 * sleep_hours + 
                   np.random.normal(0, 0.5, n_samples))
    stress_level = np.clip(stress_level, 0, 5)  # Stress scale 0-5
    
    # Previous scores have some correlation with study habits
    previous_score = (0.7 * study_hours + 0.3 * sleep_hours - 0.2 * stress_level + 
                     np.random.normal(0, 10, n_samples))
    previous_score = np.clip(previous_score, 0, 100)  # Scale 0-100
    
    # Attendance correlates with sleep and inversely with stress
    attendance_rate = (0.3 * sleep_hours - 0.4 * stress_level + 
                      np.random.normal(0.8, 0.1, n_samples))
    attendance_rate = np.clip(attendance_rate, 0, 1)  # Scale 0-1
    
    # Health status is binary, influenced by sleep and stress
    health_prob = 1 / (1 + np.exp(-(-0.3 * stress_level + 0.4 * sleep_hours)))
    health_status = (np.random.random(n_samples) < health_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'study_hours': study_hours,
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'previous_score': previous_score,
        'attendance_rate': attendance_rate,
        'health_status': health_status
    })
    
    return df

# Generate synthetic data
df = generate_synthetic_data(100)

# Normalize the features for better clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_normalized = scaler.fit_transform(df)

# Apply Bayesian Gaussian Mixture Model
n_components = 3  # Number of clusters to find
bgm = BayesianGaussianMixture(
    n_components=n_components,
    covariance_type='full',
    random_state=42,
    max_iter=1000
)

# Fit the model and get cluster assignments
df['cluster'] = bgm.fit_predict(features_normalized)

# Calculate cluster probabilities for each sample
cluster_probs = bgm.predict_proba(features_normalized)

# Add probability columns to DataFrame
for i in range(n_components):
    df[f'prob_cluster_{i}'] = cluster_probs[:, i]

# Print summary statistics for each cluster
print("\nCluster Summary Statistics:")
for cluster in range(n_components):
    print(f"\nCluster {cluster} Statistics:")
    cluster_data = df[df['cluster'] == cluster]
    print(f"Number of students: {len(cluster_data)}")
    print("\nMean values:")
    for column in df.columns[:6]:  # Original features only
        print(f"{column}: {cluster_data[column].mean():.2f}")

# Visualize the clusters
plt.figure(figsize=(15, 10))

# Create a scatter plot of study hours vs sleep hours, colored by cluster
plt.subplot(2, 2, 1)
scatter = plt.scatter(df['study_hours'], df['sleep_hours'], 
                     c=df['cluster'], cmap='viridis')
plt.xlabel('Study Hours')
plt.ylabel('Sleep Hours')
plt.title('Student Clusters: Study vs Sleep Hours')
plt.colorbar(scatter)

# Create a scatter plot of stress vs previous score
plt.subplot(2, 2, 2)
scatter = plt.scatter(df['stress_level'], df['previous_score'], 
                     c=df['cluster'], cmap='viridis')
plt.xlabel('Stress Level')
plt.ylabel('Previous Score')
plt.title('Student Clusters: Stress vs Previous Score')
plt.colorbar(scatter)

# Create a scatter plot of attendance vs health
plt.subplot(2, 2, 3)
scatter = plt.scatter(df['attendance_rate'], df['health_status'], 
                     c=df['cluster'], cmap='viridis')
plt.xlabel('Attendance Rate')
plt.ylabel('Health Status')
plt.title('Student Clusters: Attendance vs Health')
plt.colorbar(scatter)

# Create a correlation matrix plot
plt.subplot(2, 2, 4)
correlation_matrix = df.iloc[:, :6].corr()
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Feature Correlations')

plt.tight_layout()
plt.savefig('student_clusters.png')
plt.close()

# Save the dataset
df.to_csv('student_clusters.csv', index=False)

print("\nAnalysis complete! Data has been saved to 'student_clusters.csv'")
print("Visualization has been saved to 'student_clusters.png'")

# Print interesting patterns found
print("\nInteresting Patterns Found:")
print("1. Correlation between sleep and study hours:", 
      df['sleep_hours'].corr(df['study_hours']).round(3))
print("2. Correlation between stress and performance:", 
      df['stress_level'].corr(df['previous_score']).round(3))
print("3. Correlation between attendance and health:", 
      df['attendance_rate'].corr(df['health_status']).round(3))

# Print cluster characteristics
print("\nCluster Characteristics:")
for cluster in range(n_components):
    cluster_data = df[df['cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print("Average probability of assignment:", 
          cluster_probs[df['cluster'] == cluster].mean(axis=0)[cluster].round(3))
    print("Key characteristics:")
    for col in df.columns[:6]:
        mean_val = cluster_data[col].mean()
        overall_mean = df[col].mean()
        diff = mean_val - overall_mean
        if abs(diff) > 0.1 * overall_mean:  # Report if difference is >10% of mean
            print(f"- {col}: {'Higher' if diff > 0 else 'Lower'} than average "
                  f"({mean_val:.2f} vs {overall_mean:.2f})") 