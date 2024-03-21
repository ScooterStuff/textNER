import unittest
from unittest.mock import patch, MagicMock
import spacy
from spacy.tokens import Doc
from train_ner import game_entity_matcher, gesture_entity_matcher, pose_entity_matcher, train_ner

class TestNERComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a small model for testing purposes
        cls.nlp = spacy.blank("en")
    
    @patch("train_ner.spacy.load", return_value=spacy.blank("en"))
    def test_train_ner(self, mock_spacy_load):
        # Test the train_ner function
        model_dir = "./unit_test_model_dir"
        new_data = [("Minecraft is a game", {"entities": [(0, 9, "GAME")]})]
        n_iter = 1
        # Mocking the spaCy load function to return a blank English model
        # Additional mocking might be necessary depending on the train_ner function's implementation details
        train_ner(model_dir=model_dir, new_data=new_data, n_iter=n_iter)
        # This test doesn't assert anything but you should ideally check if the model was trained and saved correctly.
        # You might want to mock more parts of spaCy and check calls/arguments or use temporary directories for model saving.

if __name__ == "__main__":
    unittest.main()
