# Customer Segmentation using K-Means Clustering

This project demonstrates customer segmentation using k-means clustering on a synthetic dataset of customer purchasing behavior and demographics. The implementation helps identify distinct customer groups for targeted marketing campaigns.

## Features

- Synthetic customer data generation with realistic marketing features
- Customer segmentation based on spending patterns and demographics
- Visualization of customer clusters
- Detailed cluster analysis and profiling
- Marketing insights generation

## Dataset Features

The synthetic dataset includes the following customer characteristics:
- Total amount spent in the last year
- Average transaction value
- Monthly purchase frequency
- Product category spending ratios (electronics, clothing, groceries)
- Age
- Location (urban vs. suburban)

## Customer Segments

The implementation identifies three main customer segments:

1. **Loyal Tech Enthusiasts**
   - High spending ($5,000-$10,000/year)
   - Frequent purchases (8-12 times/month)
   - High electronics spending (70%)
   - Age 25-45
   - Mostly urban

2. **Discount Shoppers**
   - Low spending ($500-$2,000/year)
   - Infrequent purchases (1-3 times/month)
   - Low electronics spending (20%)
   - Age 40-65
   - Mostly suburban

3. **Average Young Spenders**
   - Moderate spending ($2,000-$5,000/year)
   - Regular purchases (4-7 times/month)
   - Balanced spending across categories
   - Age 18-35
   - Mix of urban and suburban

## Prerequisites

- Python 3.8+
- Required libraries (see requirements.txt)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python kmeans_example.py
```

This will:
1. Generate a synthetic customer dataset
2. Save the dataset to 'customer_segments.csv'
3. Perform k-means clustering
4. Generate visualizations
5. Save results to 'clustering_results.csv'

## Output Files

- `customer_segments.csv`: Original dataset with true customer segments
- `clustering_results.csv`: Dataset with both true and predicted segments
- `original_customer_segments.png`: Visualization of original segments
- `predicted_customer_segments.png`: Visualization of k-means clustering results

## Visualizations

The implementation generates three key visualizations:
1. Total Spending vs Purchase Frequency
2. Age vs Electronics Spending Ratio
3. Location vs Average Transaction Value

## Marketing Applications

The segmentation results can be used to:
- Create targeted marketing campaigns
- Optimize product recommendations
- Design personalized promotions
- Improve customer engagement strategies
- Allocate marketing resources effectively

## Example Output

```
Generating customer data...
Data saved to 'customer_segments.csv'

Dataset Information:
Number of customers: 1000
Number of features: 8

Cluster Analysis:
Loyal Tech Enthusiasts:
Number of customers: 333
Average total spent: $7,500.00
Average monthly frequency: 10.2
Average age: 35.2
Urban customers: 80.5%
Average spending ratios:
  Electronics: 70.5%
  Clothing: 20.2%
  Groceries: 9.3%

... 