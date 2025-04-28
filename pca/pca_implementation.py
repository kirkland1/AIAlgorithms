import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def create_customer_dataset(n_samples=100):
    """Create a synthetic dataset for customer behavior analysis"""
    np.random.seed(42)  # For reproducibility
    
    # Generate features
    age = np.random.normal(35, 10, n_samples)  # Mean age 35, std 10
    income = np.random.normal(50000, 15000, n_samples)  # Mean income $50k, std $15k
    spending = np.random.normal(1000, 300, n_samples)  # Mean spending $1000, std $300
    visits = np.random.normal(5, 2, n_samples)  # Mean visits 5, std 2
    loyalty_score = np.random.normal(75, 15, n_samples)  # Mean score 75, std 15
    
    # Create correlations between features
    spending = spending + 0.3 * income/1000  # Spending correlated with income
    visits = visits + 0.2 * loyalty_score/10  # Visits correlated with loyalty
    loyalty_score = loyalty_score + 0.4 * spending/100  # Loyalty correlated with spending
    
    # Create a DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'spending': spending,
        'visits': visits,
        'loyalty_score': loyalty_score
    })
    
    # Save to CSV
    df.to_csv('customer_dataset.csv', index=False)
    
    return df

def perform_pca(X, n_components=2):
    """Perform PCA on the dataset"""
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    return pca, pca_df, explained_variance, cumulative_variance

def plot_variance_explained(explained_variance, cumulative_variance):
    """Plot the explained variance and cumulative variance"""
    plt.figure(figsize=(10, 6))
    
    # Plot individual explained variance
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, label='Individual')
    
    # Plot cumulative explained variance
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative')
    
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend()
    plt.savefig('variance_explained.png')
    plt.show()
    plt.close()

def plot_pca_scatter(pca_df):
    """Plot the first two principal components"""
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First Two Principal Components')
    plt.savefig('pca_scatter.png')
    plt.show()
    plt.close()

def plot_feature_importance(pca, feature_names):
    """Plot feature importance in principal components"""
    # Get the loadings (feature importance) for the first two PCs
    loadings = pca.components_[:2].T
    
    # Create a DataFrame for the loadings
    loadings_df = pd.DataFrame(
        loadings,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    # Plot the loadings
    plt.figure(figsize=(10, 8))
    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, color='red', ha='center', va='center')
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.grid()
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Feature Importance in Principal Components')
    plt.savefig('feature_importance.png')
    plt.show()
    plt.close()

def main():
    # Create the dataset
    print("Creating customer behavior dataset...")
    df = create_customer_dataset(100)
    
    # Prepare features
    X = df[['age', 'income', 'spending', 'visits', 'loyalty_score']]
    
    # Perform PCA
    print("\nPerforming PCA...")
    pca, pca_df, explained_variance, cumulative_variance = perform_pca(X)
    
    # Print explained variance
    print("\nExplained Variance by Principal Components:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    print("\nCumulative Explained Variance:")
    for i, var in enumerate(cumulative_variance):
        print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
    
    # Plot variance explained
    print("\nPlotting variance explained...")
    plot_variance_explained(explained_variance, cumulative_variance)
    
    # Plot PCA scatter
    print("Plotting PCA scatter...")
    plot_pca_scatter(pca_df)
    
    # Plot feature importance
    print("Plotting feature importance...")
    plot_feature_importance(pca, X.columns)
    
    # Print feature importance in PCs
    print("\nFeature Importance in Principal Components:")
    loadings = pca.components_[:2].T
    for i, feature in enumerate(X.columns):
        print(f"{feature:15s} | PC1: {loadings[i, 0]:.3f} | PC2: {loadings[i, 1]:.3f}")

if __name__ == "__main__":
    main() 