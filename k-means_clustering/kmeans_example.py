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
    
    # Generate features with realistic distributions
    # High spenders (20% of customers)
    high_spenders = int(n_samples * 0.2)
    total_spent_high = np.random.normal(7500, 1000, high_spenders)
    frequency_high = np.random.normal(10, 1, high_spenders)
    
    # Medium spenders (50% of customers)
    medium_spenders = int(n_samples * 0.5)
    total_spent_medium = np.random.normal(3500, 500, medium_spenders)
    frequency_medium = np.random.normal(5, 1, medium_spenders)
    
    # Low spenders (30% of customers)
    low_spenders = n_samples - high_spenders - medium_spenders
    total_spent_low = np.random.normal(1500, 300, low_spenders)
    frequency_low = np.random.normal(2, 0.5, low_spenders)
    
    # Combine spending data
    total_spent = np.concatenate([total_spent_high, total_spent_medium, total_spent_low])
    frequency = np.concatenate([frequency_high, frequency_medium, frequency_low])
    
    # Calculate average transaction
    avg_transaction = total_spent / (frequency * 12)
    
    # Generate product preferences
    # Electronics preference (higher for younger customers)
    age = np.random.normal(35, 15, n_samples)
    electronics_ratio = np.clip(0.1 + (35 - age) / 100 + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    # Clothing preference (relatively stable)
    clothing_ratio = np.random.normal(0.3, 0.1, n_samples)
    
    # Groceries (remaining ratio)
    groceries_ratio = 1 - electronics_ratio - clothing_ratio
    
    # Location (urban vs suburban)
    # Higher probability of urban for higher spenders
    urban_prob = np.clip(total_spent / 10000, 0.1, 0.9)
    location = np.random.binomial(1, urban_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'total_spent': total_spent,
        'avg_transaction': avg_transaction,
        'frequency': frequency,
        'electronics_ratio': electronics_ratio,
        'clothing_ratio': clothing_ratio,
        'groceries_ratio': groceries_ratio,
        'age': age,
        'location': location
    })
    
    # Ensure ratios sum to 1
    ratio_cols = ['electronics_ratio', 'clothing_ratio', 'groceries_ratio']
    df[ratio_cols] = df[ratio_cols].div(df[ratio_cols].sum(axis=1), axis=0)
    
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
    """Analyze and interpret discovered customer segments"""
    print("\nDiscovered Customer Segments Analysis:")
    
    for cluster_id in np.unique(labels):
        cluster_data = data[labels == cluster_id]
        
        # Calculate segment characteristics
        avg_spent = cluster_data['total_spent'].mean()
        avg_freq = cluster_data['frequency'].mean()
        avg_age = cluster_data['age'].mean()
        urban_ratio = cluster_data['location'].mean()
        electronics_ratio = cluster_data['electronics_ratio'].mean()
        
        # Determine segment type based on characteristics
        if avg_spent > 6000 and avg_freq > 8:
            segment_type = "High-Value Frequent Shoppers"
        elif avg_spent < 2000 and avg_freq < 3:
            segment_type = "Budget-Conscious Infrequent Shoppers"
        elif avg_age < 30 and electronics_ratio > 0.5:
            segment_type = "Young Tech Enthusiasts"
        elif urban_ratio > 0.7 and avg_spent > 4000:
            segment_type = "Urban Premium Shoppers"
        else:
            segment_type = "General Shoppers"
        
        print(f"\nSegment {cluster_id + 1} ({segment_type}):")
        print(f"Number of customers: {len(cluster_data)}")
        print(f"Average total spent: ${avg_spent:.2f}")
        print(f"Average monthly frequency: {avg_freq:.2f}")
        print(f"Average age: {avg_age:.1f}")
        print(f"Urban customers: {urban_ratio*100:.1f}%")
        print("Average spending ratios:")
        print(f"  Electronics: {electronics_ratio*100:.1f}%")
        print(f"  Clothing: {cluster_data['clothing_ratio'].mean()*100:.1f}%")
        print(f"  Groceries: {cluster_data['groceries_ratio'].mean()*100:.1f}%")
        
        # Marketing recommendations
        print("\nMarketing Recommendations:")
        if "High-Value" in segment_type:
            print("- Premium loyalty program")
            print("- Early access to new products")
            print("- Personalized shopping experiences")
        elif "Budget-Conscious" in segment_type:
            print("- Discount and sale notifications")
            print("- Budget-friendly product recommendations")
            print("- Value bundles and packages")
        elif "Tech Enthusiasts" in segment_type:
            print("- Latest tech product announcements")
            print("- Tech accessories and upgrades")
            print("- Gaming and entertainment offers")
        elif "Urban Premium" in segment_type:
            print("- Premium urban lifestyle products")
            print("- Convenience-focused services")
            print("- Local store events and experiences")
        else:
            print("- General promotions")
            print("- Seasonal offers")
            print("- Cross-category recommendations")

def main():
    # Generate customer data
    print("Generating customer data...")
    data = generate_customer_data(n_samples=1000)
    
    # Save data to CSV
    data.to_csv('customer_data.csv', index=False)
    print("Data saved to 'customer_data.csv'")
    
    # Display data information
    print("\nDataset Information:")
    print(f"Number of customers: {len(data)}")
    print(f"Number of features: {len(data.columns)}")
    print("\nFirst 5 rows:")
    print(data.head())
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Perform k-means clustering
    print("\nPerforming k-means clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42)  # Let's try 4 clusters
    kmeans.fit(X_scaled)
    
    # Add cluster labels to data
    data['discovered_segment'] = kmeans.labels_
    
    # Plot customer clusters
    plot_customer_clusters(data, kmeans.labels_, "Discovered Customer Segments")
    
    # Analyze and interpret clusters
    analyze_clusters(data, kmeans.labels_)
    
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