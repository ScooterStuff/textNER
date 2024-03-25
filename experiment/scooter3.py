import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load your fine-tuned Sentence Transformer model
model_path = "./fine-tuned-model"  # Replace with your model path
sentence_model = SentenceTransformer(model_path)
base_words = ['kick', 'jump', 'punch', 'speak']
# Base words and their respective categories
base_words_categories = {'kick': 1, 'jump': 2, 'punch': 3, 'speak': 4}

# Synonyms for each base word (repeating the list to get 50 synonyms)
synonyms = {
    'kick': ['punt', 'strike', 'thump', 'boot', 'knock', 'whack', 'bunt', 'hit', 'tap', 'smack'],
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
# We are placing base words at the corners of a square for visualization purposes
base_word_positions = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Assign colors to each base word
colors = ['red', 'green', 'blue', 'purple']

# Function to get embeddings using SentenceTransformer
def get_embeddings(words):
    return sentence_model.encode(words)

# Generate embeddings for the base words and the new words
base_embeddings = get_embeddings(base_words)
new_embeddings = get_embeddings(new_words)

from sklearn.metrics import pairwise_distances

# # Calculate the Manhattan distances (dissimilarities) between new words and base words
# manhattan_distances = pairwise_distances(new_embeddings, base_embeddings, metric='manhattan')

# # Convert distances to similarity scores (optional, depending on your adjustment logic)
# similarity_scores = 1 / (1 + manhattan_distances)

# Calculate the Euclidean distances (dissimilarities) between new words and base words
euclidean_distances = pairwise_distances(new_embeddings, base_embeddings, metric='euclidean')

# Convert distances to similarity scores (optional, depending on your adjustment logic)
similarity_scores = 1 / (1 + euclidean_distances)

# Calculate the positions of the new words based on the similarity scores
# The position is a weighted average of the base word positions based on similarity scores
new_word_positions = similarity_scores.dot(base_word_positions) / np.sum(similarity_scores, axis=1, keepdims=True)

# Function to adjust positions towards the most similar base word
def adjust_positions_towards_base(similarity_scores, base_positions, factor=0.8):
    """
    Adjusts the position of each new word towards its most similar base word.
    :param similarity_scores: Cosine similarity scores between new words and base words.
    :param base_positions: Positions of the base words.
    :param factor: How much closer to move towards the most similar base word (0 to 1).
    :return: Adjusted positions for the new words.
    """
    adjusted_positions = np.zeros_like(new_word_positions)
    for i, scores in enumerate(similarity_scores):
        # Find the base word with the highest similarity
        most_similar_index = np.argmax(scores)
        # Calculate the adjusted position
        direction = base_positions[most_similar_index] - new_word_positions[i]
        adjusted_positions[i] = new_word_positions[i] + direction * factor
    return adjusted_positions

# Adjust the positions of the new words
adjustment_factor = 0.05  # This factor determines how much closer the new words move towards their most similar base word
adjusted_positions = adjust_positions_towards_base(similarity_scores, base_word_positions, adjustment_factor)

# Now, use 'adjusted_positions' for plotting instead of 'new_word_positions'
plt.figure(figsize=(8, 8))
for i, (word, position) in enumerate(zip(base_words, base_word_positions)):
    plt.scatter(position[0], position[1], c=colors[i], label=f'Base Word: {word}', s=100)
    plt.text(position[0], position[1], word, fontsize=12)

# Plot the adjusted positions of new words
for i, (word, position, label) in enumerate(zip(new_words, adjusted_positions, new_words_labels)):
    plt.scatter(position[0], position[1], c=colors[label-1], marker='*', s=200)
    plt.text(position[0], position[1], word, fontsize=12)

plt.title('Adjusted Word Similarity Space with SentenceTransformer')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()