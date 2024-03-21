import unittest
import json
from pathlib import Path
from motion_game_mapper import MotionGameMapper

class TestIntegrationNERModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assuming the NLP model and SentenceTransformer model are correctly loaded within the MotionGameMapper class
        cls.mapper = MotionGameMapper()

    def test_full_pipeline(self):
        # This test checks the whole pipeline using MotionGameMapper class instance
        test_sentence = "I want to play Minecraft using hand gestures, index pinch to jump."
        temp_output_file = Path("./temp_test_output.json")
        self.mapper.predict_to_json(test_sentence, str(temp_output_file))

        with open(temp_output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)

        temp_output_file.unlink()  # Clean up after test

        # Adjust these checks according to your actual expected output
        self.assertEqual(output_data["mode"], "Minecraft", "The game mode does not match.")
        self.assertTrue(any(gesture["files"] == "index_pinch" for gesture in output_data.get("gestures", [])), "The gesture 'index_pinch' was not correctly identified.")
        self.assertTrue(any(gesture["action"]["tmpt"] == "jump" for gesture in output_data.get("gestures", [])), "The action 'jump' was not correctly mapped.")

    def test_game_mode_recognition_accuracy(self):
        # Test if the system accurately recognizes and outputs the correct game mode from the provided input sentence.
        test_cases = [
            ("I want to play Minecraft with gestures.", "Minecraft"),
            ("Start a game of Roblox.", "Roblox"),
            ("I want to start Tetris today.", "Tetris"),
        ]

        for test_sentence, expected_game in test_cases:
            temp_output_file = Path(f"./system_test_game_mode_{expected_game}.json")
            
            self.mapper.predict_to_json(test_sentence, str(temp_output_file))

            with open(temp_output_file, 'r', encoding='utf-8') as file:
                output_data = json.load(file)

            temp_output_file.unlink()

            self.assertEqual(output_data["mode"], expected_game, f"Expected game mode '{expected_game}' was not recognized correctly.")

    def test_similarities_match(self):
        target_phrase = "jump"
        possible_phrases = ["run", "jump", "walk"]
        expected_match = "jump"
        match = self.mapper._similarities_match(target_phrase, possible_phrases, self.mapper.sentence_model)
        self.assertEqual(match, expected_match, "The _similarities_match method did not return the expected most similar phrase.")

if __name__ == "__main__":
    unittest.main()
