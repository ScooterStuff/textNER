import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Function to get embeddings
def get_embeddings(words, model):
    return model.encode(words)

# Load the Sentence Transformer model
model_name = 'bert-base-nli-mean-tokens'  # Example model, replace with your actual model
model = SentenceTransformer(model_name)

# Base words and their synonyms
base_words = ['jump', 'punch', 'speak']
synonyms = {
    'jump': ['leap', 'hop', 'bounce', 'vault', 'spring', 'bound', 'skip', 'dive', 'launch', 'hurdle'],
    'punch': ['jab', 'strike', 'thrust', 'swing', 'hook', 'uppercut', 'slam', 'smash', 'bash', 'beat'],
    'speak': ['talk', 'utter', 'tell', 'disclose', 'reveal', 'express', 'say', 'articulate', 'pronounce', 'voice']
}

# Preparing the data
new_words = [word for sublist in synonyms.values() for word in sublist]
new_words_labels = np.concatenate([[i] * len(synonyms[base_words[i]]) for i in range(len(base_words))])

# Generate embeddings for base and new words
base_embeddings = get_embeddings(base_words, model)
new_embeddings = get_embeddings(new_words, model)

# Calculate cosine similarity scores
similarity_scores = cosine_similarity(new_embeddings, base_embeddings)

# Calculate positions in 3D space based on similarity scores
base_word_positions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # X, Y, Z axis respectively
scaling_factors = [2, 2, 2]  # Example scaling factors for each axis
new_word_positions = np.zeros((len(new_words), 3))

for i, scores in enumerate(similarity_scores):
    most_similar_base_idx = np.argmax(scores)
    # Scale the position towards the most similar base word's axis
    for j in range(3):
        if j == most_similar_base_idx:
            new_word_positions[i, j] = scores[j] * scaling_factors[j]
        else:
            new_word_positions[i, j] = scores[j]

# Perform KMeans clustering in 3D
kmeans = KMeans(n_clusters=len(base_words), random_state=0).fit(new_word_positions)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 3D plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['green', 'blue', 'purple']
for i, word in enumerate(base_words):
    ax.scatter(*base_word_positions[i], c=colors[i], label=f'Base Word: {word}', s=100)
    ax.text(*base_word_positions[i], word, fontsize=12)

for i, word in enumerate(new_words):
    ax.scatter(*new_word_positions[i], c=colors[cluster_labels[i]], marker='*', s=200)
    ax.text(*new_word_positions[i], word, fontsize=10)

for centroid, color in zip(centroids, colors):
    ax.scatter(*centroid, c=color, marker='x', s=200, lw=2)

ax.set_xlabel('X - Punch')
ax.set_ylabel('Y - Jump')
ax.set_zlabel('Z - Speak')
plt.title('3D Visualization of Word Embeddings with Adjustments')
plt.legend()
plt.show()
