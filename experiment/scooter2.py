import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Random seed for reproducibility
np.random.seed(42)

# Function to simulate word embeddings
def generate_random_embedding():
    return np.random.rand(300)

# Define a simple structure to hold our word and its category
class WordEmbedding:
    def __init__(self, word, category):
        self.word = word
        self.category = category
        self.embedding = generate_random_embedding()

# Generate embeddings for our base words
base_words = [
    WordEmbedding('punch', 1),
    WordEmbedding('kick', 2),
    WordEmbedding('jump', 3)
]

# Generate embeddings for words similar to our base words
# Adding 10 synonyms for each category
synonyms_per_category = {
    1: ['hit', 'jab', 'strike', 'hook', 'uppercut', 'cross', 'slap', 'thump', 'bash', 'swat'],
    2: ['thrust', 'thump', 'punt', 'smash', 'whack', 'knock', 'bop', 'slam', 'bump', 'wallop'],
    3: ['leap', 'hop', 'spring', 'vault', 'bounce', 'skip', 'bound', 'hurdle', 'lunge', 'dash']
}

similar_words = []
for category, words in synonyms_per_category.items():
    for word in words:
        similar_words.append(WordEmbedding(word, category))

# Combine base words and similar words for clustering
all_words = base_words + similar_words

# Extract embeddings and labels
embeddings = np.array([word.embedding for word in all_words])
labels = np.array([word.category for word in all_words])

# Calculate pairwise cosine similarity (distance)
cosine_dist_matrix = cosine_distances(embeddings)

# Perform KMeans clustering based on cosine similarity with n_init set explicitly
kmeans_cosine = KMeans(n_clusters=3, n_init=10, random_state=42).fit(cosine_dist_matrix)

# Use PCA to reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot the PCA-reduced embeddings
plt.figure(figsize=(10, 10))

# Scatter plot for base words and synonyms
for category, marker, color in zip(range(1, 4), ('o', 's', '^'), ('blue', 'green', 'red')):
    # Plot base words
    plt.scatter(reduced_embeddings[labels == category, 0], 
                reduced_embeddings[labels == category, 1], 
                c=color, 
                label=f'Base Word Category {category}', 
                marker=marker,
                s=100)

    # Plot similar words
    plt.scatter(reduced_embeddings[labels == category, 0], 
                reduced_embeddings[labels == category, 1], 
                c=color, 
                label=f'Similar Word Category {category}', 
                marker='*',
                s=50)

# Annotate points with the word names
for i, word in enumerate(all_words):
    plt.annotate(word.word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8)

# Add legend and labels
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Word Clusters (Cosine Similarity)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
