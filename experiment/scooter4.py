import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load your fine-tuned Sentence Transformer model
model_path = "./fine-tuned-model"  # Replace with your model path
sentence_model = SentenceTransformer(model_path)
model_name = 'bert-base-nli-mean-tokens'  # Choose from the models above
model = SentenceTransformer(model_name)
base_words = ['jump', 'punch', 'speak']
# Base words and their respective categories
base_words_categories = {'jump': 1, 'punch': 2, 'speak': 3}

# Synonyms for each base word (repeating the list to get 50 synonyms)
synonyms = {
    'jump': ['leap', 'hop', 'bounce', 'vault', 'spring', 'bound', 'skip', 'dive', 'launch', 'hurdle'],
    'punch': ['jab', 'strike', 'thrust', 'swing', 'hook', 'uppercut', 'slam', 'smash', 'bash', 'beat'],
    'speak': ['talk', 'utter', 'tell', 'disclose', 'reveal', 'express', 'say', 'articulate', 'pronounce', 'voice']
}

# Flatten the list of synonyms and assign labels
new_words = []
new_words_labels = []

for base_word, syn_list in synonyms.items():
    new_words.extend(syn_list)
    new_words_labels.extend([base_words_categories[base_word]] * len(syn_list))

# Convert to numpy arrays
new_words = np.array(new_words)
new_words_labels = np.array(new_words_labels)
# We are placing base words at the corners of an equilateral triangle for visualization purposes
base_word_positions = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

# Assign colors to each base word
colors = ['green', 'blue', 'purple']

# Function to get embeddings using SentenceTransformer
def get_embeddings(words):
    return sentence_model.encode(words)

# Generate embeddings for the base words and the new words
base_embeddings = get_embeddings(base_words)
new_embeddings = get_embeddings(new_words)

from sklearn.metrics import pairwise_distances

manhattan_distances = pairwise_distances(new_embeddings, base_embeddings, metric='manhattan')
similarity_scores = 1 / (1 + manhattan_distances)

# euclidean_distances = pairwise_distances(new_embeddings, base_embeddings, metric='euclidean')
# similarity_scores = 1 / (1 + euclidean_distances)

# similarity_scores = cosine_similarity(new_embeddings, base_embeddings)

# Calculate the positions of the new words based on the similarity scores
# The position is a weighted average of the base word positions based on similarity scores
new_word_positions = similarity_scores.dot(base_word_positions) / np.sum(similarity_scores, axis=1, keepdims=True)

# Function to adjust positions towards the most similar base word
# Function to adjust positions towards the two most similar base words
def adjust_positions_towards_two_bases(similarity_scores, base_positions, first_factor=0.8, second_factor=0.2):
    """
    Adjusts the position of each new word towards its two most similar base words.
    The first most similar base word has a greater influence as defined by first_factor.
    The second most similar base word has less influence as defined by second_factor.
    :param similarity_scores: Cosine similarity scores between new words and base words.
    :param base_positions: Positions of the base words.
    :param first_factor: Influence factor for the most similar base word.
    :param second_factor: Influence factor for the second most similar base word.
    :return: Adjusted positions for the new words.
    """
    adjusted_positions = np.zeros_like(new_word_positions)
    for i, scores in enumerate(similarity_scores):
        # Find indices of the two highest similarity scores
        first_most_similar_index = np.argmax(scores)
        second_most_similar_index = np.argsort(scores)[-2]  # The second largest value
        # Calculate the adjusted position
        first_direction = base_positions[first_most_similar_index] - new_word_positions[i]
        second_direction = base_positions[second_most_similar_index] - new_word_positions[i]
        adjusted_positions[i] = new_word_positions[i] + first_direction * first_factor + second_direction * second_factor
    return adjusted_positions

# Adjust the positions of the new words using the new function
adjustment_factor_first = 0.2  # Strong influence of the most similar base word
adjustment_factor_second = 0.05  # Less influence of the second most similar base word
adjusted_positions = adjust_positions_towards_two_bases(similarity_scores, base_word_positions, adjustment_factor_first, adjustment_factor_second)

from sklearn.cluster import KMeans

# Number of clusters
k = 3  # Assuming you want to cluster into the same number of categories as base words

# Initialize the KMeans object
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the model to the adjusted positions
kmeans.fit(adjusted_positions)

# Get the cluster assignments for each word
cluster_labels = kmeans.labels_

# Get the coordinates of cluster centers
centroids = kmeans.cluster_centers_

# Plot the results
plt.figure(figsize=(8, 8))
for i, (word, position) in enumerate(zip(base_words, base_word_positions)):
    plt.scatter(position[0], position[1], c=colors[i], label=f'Base Word: {word}', s=100)
    plt.text(position[0], position[1], word, fontsize=12)

# Plot the adjusted positions of new words with cluster colors
for i, (word, position, label) in enumerate(zip(new_words, adjusted_positions, cluster_labels)):
    # if colors[label] == 'purple':
    #     plt.scatter(position[0], position[1], c='blue', marker='*', s=200)
    #     plt.text(position[0], position[1], word, fontsize=12)
    # elif colors[label] == 'blue':
    #     plt.scatter(position[0], position[1], c='purple', marker='*', s=200)
    #     plt.text(position[0], position[1], word, fontsize=12)
    # else:
    #     plt.scatter(position[0], position[1], c=colors[label], marker='*', s=200)
    #     plt.text(position[0], position[1], word, fontsize=12)
    plt.scatter(position[0], position[1], c=colors[label], marker='*', s=200)
    plt.text(position[0], position[1], word, fontsize=12)
    

# Plot centroids
for centroid, color in zip(centroids, colors):
    # if color == 'purple':
    #     plt.scatter(centroid[0], centroid[1], c='blue', marker='x', s=200, lw=2)
    # elif color == 'blue':
    #     plt.scatter(centroid[0], centroid[1], c='purple', marker='x', s=200, lw=2)
    # else:
    #     plt.scatter(centroid[0], centroid[1], c=color, marker='x', s=200, lw=2)
    plt.scatter(centroid[0], centroid[1], c=color, marker='x', s=200, lw=2)
    

plt.title('K-Means Clustering on Adjusted Word Similarity Space (Manhattan)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.legend()
plt.show()
