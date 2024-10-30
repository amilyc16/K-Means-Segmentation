# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Simulate a customer transaction dataset
np.random.seed(42)  # For reproducibility

# Create a DataFrame with sample data
num_samples = 100
data = pd.DataFrame({
    'student_id': np.arange(1, num_samples + 1),  # Unique student IDs from 1 to 100
    'dining_hall': np.random.choice(['Hall A', 'Hall B', 'Hall C'], num_samples),  # Random dining halls
    'money_spent': np.random.uniform(5.0, 25.0, num_samples),  # Random spending between $5 and $25
    'meal_type': np.random.choice(['Breakfast', 'Lunch', 'Dinner'], num_samples),  # Random meal types
    'visit_time': np.random.choice(['Morning', 'Afternoon', 'Evening'], num_samples),  # Random visit times
    'visit_frequency': np.random.randint(1, 16, num_samples)  # Random frequency between 1 and 15 visits per week
})

# Standardize the relevant features for clustering
features_to_scale = ['money_spent', 'visit_frequency']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features_to_scale])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(data_scaled)

# Visualize Average Money Spent and Visit Frequency by Dining Hall and Cluster
avg_data = data.groupby(['dining_hall', 'cluster']).agg({'money_spent': 'mean', 'visit_frequency': 'mean'}).reset_index()

# Create a bar plot for Average Visit Frequency by Dining Hall and Cluster
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_data, x='dining_hall', y='visit_frequency', hue='cluster', palette='viridis')
plt.title('Average Visit Frequency by Dining Hall and Cluster')
plt.ylabel('Average Visit Frequency (per week)')
plt.xlabel('Dining Hall')
plt.legend(title='Cluster')
plt.show()
