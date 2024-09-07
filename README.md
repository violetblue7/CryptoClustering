# CryptoClustering
Module 19

**CODE EXPLANATION**

**1. Prepare the Data**
**Normalize Data:** Use StandardScaler from scikit-learn to standardize features (mean=0, variance=1).
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['feature1', 'feature2', ...]])

**2.Create DataFrame:** Convert the scaled data into a DataFrame and set the "coin_id" as the index.

scaled_df = pd.DataFrame(scaled_data, columns=['feature1', 'feature2', ...], index=df['coin_id'])
Check Data: Display the first five rows of the scaled DataFrame.

print(scaled_df.head())

**2. Find the Best Value for k Using the Original Scaled Data**
**2.1 List of k Values:** Create a range of k values (from 1 to 11).

k_values = range(1, 12)

**2.2 Compute Inertia:** Calculate the inertia (within-cluster sum of squares) for each k value.

from sklearn.cluster import KMeans

inertia = []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

**2.3 Plot Elbow Curve:** Plot inertia values to visualize the elbow and determine the best k.

import matplotlib.pyplot as plt

plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

**2.4 Determine Optimal k:** Identify the value of k where the inertia starts to decrease more slowly (the "elbow" point).

**3. Cluster Cryptocurrencies with K-means Using the Original Scaled Data**

**3.1 Initialize K-means:** Set up K-means with the optimal k.

optimal_k = 3  # Replace with your best k value
kmeans = KMeans(n_clusters=optimal_k)

**3.2 Fit and Predict:** Fit the K-means model and predict clusters.

clusters = kmeans.fit_predict(scaled_df)
scaled_df['cluster'] = clusters

**3.3 Plot Data:** Use hvPlot tocreate a scatter plot with price_change_percentage_24h vs. price_change_percentage_7d, color-coded by clusters.

import hvplot.pandas

scatter_plot = scaled_df.hvplot.scatter(
    x='price_change_percentage_24h',
    y='price_change_percentage_7d',
    c='cluster',
    hover_cols=['coin_id'],
    title='Cryptocurrency Clusters'
)
scatter_plot

**4. Optimize Clusters with Principal Component Analysis (PCA)**
**4.1 Perform PCA:** Reduce the feature set to three principal components.

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)
**4.2 Explained Variance:** Retrieve and sum the explained variance ratios.

explained_variance = pca.explained_variance_ratio_.sum()
print(f'Total explained variance: {explained_variance}')

**4.3 Create PCA DataFrame:** Convert PCA data to DataFrame and set "coin_id" as the index.

pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'], index=df['coin_id'])

**5. Find the Best Value for k Using the PCA Data**
**5.1Repeat Elbow Method:** Follow the same steps as in step 2 but with PCA-transformed data.

k_values = range(1, 12)
inertia_pca = []
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca_df)
    inertia_pca.append(kmeans.inertia_)

plt.plot(k_values, inertia_pca, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for PCA Data')
plt.show()

**5.2 Compare k Values:** 
Note if the optimal k differs from the original data.

**6. Cluster Cryptocurrencies with K-means Using the PCA Data**
Initialize K-means: Use the optimal k value from PCA data.

kmeans_pca = KMeans(n_clusters=optimal_k_pca)

**6.1 Fit and Predict: Fit the model and predict clusters for PCA data.**

clusters_pca = kmeans_pca.fit_predict(pca_df)
pca_df['cluster'] = clusters_pca

**6.2 Plot PCA Data:** Create a scatter plot with PCA components.
Plot PCA Data: Create a scatter plot with PCA components.

scatter_plot_pca = pca_df.hvplot.scatter(
    x='PC1',
    y='PC2',
    c='cluster',
    hover_cols=['coin_id'],
    title='Cryptocurrency Clusters (PCA Data)'
)
scatter_plot_pca

**REFERENCES**
https://bootcampspot.instructure.com/courses/6446/external_tools/313#:~:text=Tutoring%20Sessions-,Xpert%20Learning%20Assistant,-Lucid 

StandardScaler and DataFrame Creation:
StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html 

Elbow Method: https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/#:~:text=This%20method%20is%20a%20visual,graph%20bends%20like%20an%20elbow. 

K-Means Clustering: https://builtin.com/data-science/elbow-method#:~:text=The%20elbow%20method%20is%20a%20graphical%20method%20for%20finding%20the,the%20graph%20forms%20an%20elbow. 

HoloViews: https://holoviews.org/
