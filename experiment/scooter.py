import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Data Preparation
# Simulate embeddings for 1000 words
np.random.seed(42)  # For reproducibility
word_embeddings = np.random.rand(1000, 300)  # 1000 words, 300-dimensional embeddings

# Step 2: Similarity Calculation
# Calculate the distance matrices for each similarity measure
cosine_distances = pairwise_distances(word_embeddings, metric='cosine')
euclidean_distances = pairwise_distances(word_embeddings, metric='euclidean')
manhattan_distances = pairwise_distances(word_embeddings, metric='manhattan')

# Step 3: Clustering
# Perform K-means clustering on each distance matrix
kmeans_cosine = KMeans(n_clusters=3, random_state=42).fit(cosine_distances)
kmeans_euclidean = KMeans(n_clusters=3, random_state=42).fit(euclidean_distances)
kmeans_manhattan = KMeans(n_clusters=3, random_state=42).fit(manhattan_distances)

# Step 4: Calculate Silhouette Scores for each clustering result
silhouette_cosine = silhouette_score(cosine_distances, kmeans_cosine.labels_, metric="precomputed")
silhouette_euclidean = silhouette_score(euclidean_distances, kmeans_euclidean.labels_, metric="precomputed")
silhouette_manhattan = silhouette_score(manhattan_distances, kmeans_manhattan.labels_, metric="precomputed")

print("Silhouette Scores:")
print("Cosine:", silhouette_cosine)
print("Euclidean:", silhouette_euclidean)
print("Manhattan:", silhouette_manhattan)

# Visualization of clusters (using Cosine for example)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(word_embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans_cosine.labels_, cmap='viridis', marker='o', edgecolor='k', s=150)
plt.title('Word Clusters (Cosine Similarity)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()
