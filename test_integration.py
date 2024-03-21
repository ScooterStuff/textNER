import unittest
import json
from pathlib import Path
import spacy
from predict_org import predict_to_json, similarties_match

class TestIntegrationNERModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assuming the NLP model is correctly loaded with necessary matchers in your script
        cls.nlp_model_path = "./ner_model"
        cls.nlp = spacy.load(cls.nlp_model_path)

    def test_full_pipeline(self):
        # This test assumes your NER and action mapping works as expected and checks the whole pipeline
        test_sentence = "I want to play Minecraft using hand gestures, index pinch to jump."
        temp_output_file = Path("./temp_test_output.json")
        predict_to_json(test_sentence, str(temp_output_file))

        with open(temp_output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)

        temp_output_file.unlink()  # Clean up after test

        # Adjust these checks according to your actual expected output
        self.assertEqual(output_data["mode"], "Minecraft", "The game mode does not match.")
        self.assertTrue(any(gesture["files"] == "index_pinch" for gesture in output_data.get("gestures", [])), "The gesture 'pinch' was not correctly identified.")
        self.assertTrue(any(gesture["action"]["tmpt"] == "jump" for gesture in output_data.get("gestures", [])), "The action 'jump' was not correctly mapped.")
    def test_predict_to_json_output_structure(self):
        """
        Test if the predict_to_json function generates the correct output structure
        for a given input.
        """
        test_sentence = "I want to play Tetris with my left hand, Rotate item using hadouken."
        expected_structure_keys = ["mode", "orientation", "landmark", "poses", "gestures"]
        temp_output_file = Path("./system_test_output.json")
        
        # Generate JSON output
        predict_to_json(test_sentence, str(temp_output_file))

        # Load the generated JSON
        with open(temp_output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)
        
        # Clean up after test
        temp_output_file.unlink()

        # Check if all expected keys are in the output structure
        for key in expected_structure_keys:
            self.assertIn(key, output_data, f"Key '{key}' missing in output JSON structure.")
        
    def test_game_mode_recognition_accuracy(self):
        """
        Test if the system accurately recognizes and outputs the correct game mode
        from the provided input sentence.
        """
        test_cases = [
            ("I want to play Minecraft with gestures.", "Minecraft"),
            ("Start a game of Roblox.", "Roblox"),
            ("I want to start Tetris today.", "Tetris"),
        ]
        
        for test_sentence, expected_game in test_cases:
            temp_output_file = Path(f"./system_test_game_mode_{expected_game}.json")
            
            predict_to_json(test_sentence, str(temp_output_file))

            with open(temp_output_file, 'r', encoding='utf-8') as file:
                output_data = json.load(file)

            temp_output_file.unlink()

            self.assertEqual(output_data["mode"], expected_game, f"Expected game mode '{expected_game}' was not recognized correctly.")

    def test_similarities_match(self):
        target_phrase = "jump"
        possible_phrases = ["run", "jump", "walk"]
        expected_match = "jump"
        match = similarties_match(target_phrase, possible_phrases)
        self.assertEqual(match, expected_match, "The similarities_match function did not return the expected most similar phrase.")


if __name__ == "__main__":
    unittest.main()
