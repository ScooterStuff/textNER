from spacy.util import filter_spans
import nltk
from nltk.corpus import wordnet
import random
from itertools import chain
from train_data import TRAIN_DATA

nltk.download('wordnet', quiet=True)

class DataAugmenter:
    """
    A class for augmenting NLP training data using synonym replacement.
    """

    def __init__(self, train_data):
        """
        Initializes the DataAugmenter with the training data.

        :param train_data: The initial training data.
        """
        self.train_data = train_data

    def synonym_replacement(self, text, entity_spans):
        """
        Replaces words in the given text with their synonyms.

        :param text: The text to process.
        :param entity_spans: The spans of entities in the text.
        :return: A tuple of the new text and updated entity spans.
        """
        words = text.split()
        new_words = words.copy()
        changes = {}  # Track changes in word lengths

        for i, word in enumerate(words):
            synonyms = set(chain.from_iterable([syn.lemma_names() for syn in wordnet.synsets(word)]))
            synonyms.discard(word.lower())  # Discard the original word to avoid redundancy
            if synonyms:
                synonym = random.choice(list(synonyms))
                changes[i] = (len(word), len(synonym))  # Original and new length
                new_words[i] = synonym

        new_text = ' '.join(new_words)
        new_spans = self.adjust_entity_spans(text, new_text, entity_spans)

        return new_text, new_spans

    def augment_data_with_synonyms(self):
        """
        Augments the training data with synonym replacements.

        :return: The augmented training data.
        """
        augmented_data = []
        for text, annotations in self.train_data:
            entity_spans = [(start, end, label) for start, end, label in annotations['entities']]
            augmented_text, new_entity_spans = self.synonym_replacement(text, entity_spans)
            new_annotations = {"entities": new_entity_spans}
            augmented_data.append((augmented_text, new_annotations))
        return augmented_data + self.train_data

    @staticmethod
    def adjust_entity_spans(original_text, new_text, original_spans):
        """
        Adjusts the entity spans in the new text.

        :param original_text: The original text.
        :param new_text: The new text with synonyms replaced.
        :param original_spans: The original entity spans.
        :return: The adjusted entity spans.
        """
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, original_text, new_text)
        new_spans = []

        for opcode in matcher.get_opcodes():
            tag, i1, i2, j1, j2 = opcode
            if tag == 'equal':
                shift = j1 - i1
                for start, end, label in original_spans:
                    if start >= i1 and end <= i2:
                        new_start = start + shift
                        new_end = end + shift
                        new_spans.append((new_start, new_end, label))
        return new_spans

# Example usage:
if __name__ == "__main__":
    # Assuming TRAIN_DATA is defined elsewhere and imported
    augmenter = DataAugmenter(TRAIN_DATA)
    augmented_data = augmenter.augment_data_with_synonyms()
    print(augmented_data)
