import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def generate_customer_data(n_samples=1000):
    """Generate synthetic customer data with realistic marketing features"""
    np.random.seed(42)
    
    # Define cluster characteristics
    cluster_profiles = {
        'Loyal Tech Enthusiasts': {
            'spending_range': (5000, 10000),
            'frequency_range': (8, 12),
            'electronics_ratio': 0.7,
            'age_range': (25, 45),
            'urban_ratio': 0.8
        },
        'Discount Shoppers': {
            'spending_range': (500, 2000),
            'frequency_range': (1, 3),
            'electronics_ratio': 0.2,
            'age_range': (40, 65),
            'urban_ratio': 0.3
        },
        'Average Young Spenders': {
            'spending_range': (2000, 5000),
            'frequency_range': (4, 7),
            'electronics_ratio': 0.4,
            'age_range': (18, 35),
            'urban_ratio': 0.6
        }
    }
    
    data = []
    labels = []
    
    for cluster_name, profile in cluster_profiles.items():
        n_cluster_samples = n_samples // len(cluster_profiles)
        
        # Generate features based on cluster profile
        total_spent = np.random.uniform(
            profile['spending_range'][0],
            profile['spending_range'][1],
            n_cluster_samples
        )
        
        frequency = np.random.uniform(
            profile['frequency_range'][0],
            profile['frequency_range'][1],
            n_cluster_samples
        )
        
        avg_transaction = total_spent / (frequency * 12)  # Monthly average
        
        # Generate product category spending ratios
        electronics = np.random.normal(
            profile['electronics_ratio'],
            0.1,
            n_cluster_samples
        )
        clothing = np.random.normal(0.3, 0.1, n_cluster_samples)
        groceries = 1 - electronics - clothing
        
        # Generate demographic features
        age = np.random.uniform(
            profile['age_range'][0],
            profile['age_range'][1],
            n_cluster_samples
        )
        
        # Location (1 for urban, 0 for suburban)
        location = np.random.binomial(1, profile['urban_ratio'], n_cluster_samples)
        
        # Combine features
        cluster_data = np.column_stack((
            total_spent,
            avg_transaction,
            frequency,
            electronics,
            clothing,
            groceries,
            age,
            location
        ))
        
        data.append(cluster_data)
        labels.extend([cluster_name] * n_cluster_samples)
    
    # Combine all clusters
    X = np.vstack(data)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[
        'total_spent',
        'avg_transaction',
        'frequency',
        'electronics_ratio',
        'clothing_ratio',
        'groceries_ratio',
        'age',
        'location'
    ])
    
    # Add noise to make it more realistic
    for col in df.columns:
        if col != 'location':  # Don't add noise to binary location
            df[col] += np.random.normal(0, df[col].std() * 0.1, len(df))
    
    # Ensure ratios sum to 1
    ratio_cols = ['electronics_ratio', 'clothing_ratio', 'groceries_ratio']
    df[ratio_cols] = df[ratio_cols].div(df[ratio_cols].sum(axis=1), axis=0)
    
    # Add cluster labels
    df['customer_segment'] = labels
    
    return df

def plot_customer_clusters(data, labels, title):
    """Plot customer clusters using key features"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Spending vs Frequency
    plt.subplot(2, 2, 1)
    sns.scatterplot(
        x='total_spent',
        y='frequency',
        hue=labels,
        data=data,
        palette='viridis',
        alpha=0.7
    )
    plt.title('Total Spending vs Purchase Frequency')
    plt.xlabel('Total Amount Spent ($)')
    plt.ylabel('Monthly Purchase Frequency')
    
    # Plot 2: Age vs Electronics Ratio
    plt.subplot(2, 2, 2)
    sns.scatterplot(
        x='age',
        y='electronics_ratio',
        hue=labels,
        data=data,
        palette='viridis',
        alpha=0.7
    )
    plt.title('Age vs Electronics Spending Ratio')
    plt.xlabel('Age')
    plt.ylabel('Electronics Spending Ratio')
    
    # Plot 3: Location vs Average Transaction
    plt.subplot(2, 2, 3)
    sns.boxplot(
        x='location',
        y='avg_transaction',
        hue=labels,
        data=data,
        palette='viridis'
    )
    plt.title('Location vs Average Transaction Value')
    plt.xlabel('Location (0: Suburban, 1: Urban)')
    plt.ylabel('Average Transaction Value ($)')
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def analyze_clusters(data, labels):
    """Analyze and print cluster characteristics"""
    print("\nCluster Analysis:")
    for cluster in data['customer_segment'].unique():
        cluster_data = data[data['customer_segment'] == cluster]
        print(f"\n{cluster}:")
        print(f"Number of customers: {len(cluster_data)}")
        print(f"Average total spent: ${cluster_data['total_spent'].mean():.2f}")
        print(f"Average monthly frequency: {cluster_data['frequency'].mean():.2f}")
        print(f"Average age: {cluster_data['age'].mean():.1f}")
        print(f"Urban customers: {cluster_data['location'].mean()*100:.1f}%")
        print("Average spending ratios:")
        print(f"  Electronics: {cluster_data['electronics_ratio'].mean()*100:.1f}%")
        print(f"  Clothing: {cluster_data['clothing_ratio'].mean()*100:.1f}%")
        print(f"  Groceries: {cluster_data['groceries_ratio'].mean()*100:.1f}%")

def main():
    # Generate customer data
    print("Generating customer data...")
    data = generate_customer_data(n_samples=1000)
    
    # Save data to CSV
    data.to_csv('customer_segments.csv', index=False)
    print("Data saved to 'customer_segments.csv'")
    
    # Display data information
    print("\nDataset Information:")
    print(f"Number of customers: {len(data)}")
    print(f"Number of features: {len(data.columns) - 1}")  # Excluding segment column
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Prepare data for clustering
    X = data.drop('customer_segment', axis=1)
    true_labels = data['customer_segment']
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Plot original clusters
    plot_customer_clusters(data, true_labels, "Original Customer Segments")
    
    # Perform k-means clustering
    print("\nPerforming k-means clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    
    # Add cluster labels to data
    data['predicted_segment'] = kmeans.labels_
    
    # Plot predicted clusters
    plot_customer_clusters(data, kmeans.labels_, "Predicted Customer Segments")
    
    # Analyze clusters
    analyze_clusters(data, true_labels)
    
    # Evaluate clustering
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    print(f"\nClustering Evaluation:")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    # Save results
    data.to_csv('clustering_results.csv', index=False)
    print("\nResults saved to 'clustering_results.csv'")

if __name__ == "__main__":
    main() 