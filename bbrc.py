import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load your Sentence Transformer model
model_name = 'bert-base-nli-mean-tokens'  # Example model, replace with your actual model
model = SentenceTransformer(model_name)

base_words = ['jump', 'punch', 'speak']
base_words_categories = {'jump': 1, 'punch': 2, 'speak': 3}

# Synonyms for each base word
synonyms = {
    'jump': ['leap', 'hop', 'bounce', 'vault', 'spring', 'bound', 'skip', 'dive', 'launch', 'hurdle'],
    'punch': ['jab', 'strike', 'thrust', 'swing', 'hook', 'uppercut', 'slam', 'smash', 'bash', 'beat'],
    'speak': ['talk', 'utter', 'tell', 'disclose', 'reveal', 'express', 'say', 'articulate', 'pronounce', 'voice']
}

new_words = []
new_words_labels = []

for base_word, syn_list in synonyms.items():
    new_words.extend(syn_list)
    new_words_labels.extend([base_words_categories[base_word]] * len(syn_list))

new_words = np.array(new_words)
new_words_labels = np.array(new_words_labels)

# Assign base words to the corners of a 3D space
base_word_positions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # X, Y, Z axis respectively

colors = ['green', 'blue', 'purple']

# Function to get embeddings
def get_embeddings(words):
    return model.encode(words)

# Generate embeddings
base_embeddings = get_embeddings(base_words)
new_embeddings = get_embeddings(new_words)

# Calculate distances (using Euclidean for this example)
similarity_scores = cosine_similarity(new_embeddings, base_embeddings)

# Calculate positions in 3D space based on similarity scores
new_word_positions = similarity_scores.dot(base_word_positions) / np.sum(similarity_scores, axis=1, keepdims=True)

# KMeans clustering in 3D
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(new_word_positions)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 3D Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot base words
for i, (word, position) in enumerate(zip(base_words, base_word_positions)):
    ax.scatter(position[0], position[1], position[2], c=colors[i], label=f'Base Word: {word}', s=100)
    ax.text(position[0], position[1], position[2], word, fontsize=12)

# Plot new words
for i, (word, position, label) in enumerate(zip(new_words, new_word_positions, cluster_labels)):
    ax.scatter(position[0], position[1], position[2], c=colors[label], marker='*', s=200)
    ax.text(position[0], position[1], position[2], word, fontsize=10)

# Plot centroids
for centroid, color in zip(centroids, colors):
    ax.scatter(centroid[0], centroid[1], centroid[2], c=color, marker='x', s=200, lw=2)

ax.set_xlabel('X - Punch')
ax.set_ylabel('Y - Jump')
ax.set_zlabel('Z - Speak')
plt.title('3D Visualization of Word Embeddings and Clusters')
plt.legend()
plt.show()
