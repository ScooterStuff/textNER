import unittest
import json
from pathlib import Path
from spacy.util import compile_infix_regex
from predict import predict_to_json
import spacy

class TestIntegrationNERModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the NLP model, assuming it's already trained and saved
        cls.nlp = spacy.load("./ner_model")

        # If your predict.py uses a specific NLP model, ensure it's loaded here
        # cls.nlp.add_pipe("game_entity_matcher")
        # cls.nlp.add_pipe("gesture_entity_matcher")
        # cls.nlp.add_pipe("pose_entity_matcher")

    def test_full_pipeline(self):
        test_sentence = "I want to play Minecraft using hand gestures, specifically pinch to jump."
        expected_output = {
            "mode": "Minecraft",
            "poses": [],
            "gestures": [{"files": "pinch", "action": {"tmpt": "jump", "class": "", "method": "click", "args": []}}],
        }

        # Temporary output file for testing
        temp_output_file = Path("./temp_test_output.json")
        predict_to_json(test_sentence, str(temp_output_file))

        with open(temp_output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)

        # Clean up the temporary file
        temp_output_file.unlink()

        self.assertEqual(output_data["mode"], expected_output["mode"], "The game mode does not match.")
        self.assertTrue(any(gesture["files"] == "pinch" for gesture in output_data["gestures"]), "The gesture does not match.")
        self.assertTrue(any(gesture["action"]["tmpt"] == "jump" for gesture in output_data["gestures"]), "The action does not match.")



class TestModelAccuracy(unittest.TestCase):
    def test_combined_entity_recognition(self):
        test_texts = [
            "I want to play Final Fantasy XV with my feet.",
            "Shoot using two fingers in Call of Duty.",
        ]
        expected_entities = [
            ("Final Fantasy XV", "GAME"),
            ("two fingers", "POSE"),
        ]

        for test_text, (expected_text, expected_label) in zip(test_texts, expected_entities):
            doc = TestIntegrationNERModel.nlp(test_text)
            found_entities = [(ent.text, ent.label_) for ent in doc.ents]

            self.assertIn((expected_text, expected_label), found_entities, f"Entity {expected_text} with label {expected_label} not found in text: {test_text}")

class TestComplexCommandJSONCreation(unittest.TestCase):
    def test_complex_command_json_output(self):
        test_sentence = "In FIFA, use my right hand to control the player, fist to shoot, and swipe to pass."
        expected_actions = ["shoot", "pass"]  # Simplified for demonstration

        temp_output_file = Path("./complex_command_output.json")
        predict_to_json(test_sentence, str(temp_output_file))

        with open(temp_output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)

        temp_output_file.unlink()  # Clean up

        # Check if all expected actions are present in either gestures or poses
        actions_from_output = [action["action"]["tmpt"] for action in output_data["gestures"] + output_data["poses"]]
        for expected_action in expected_actions:
            self.assertIn(expected_action, actions_from_output, f"Expected action {expected_action} not found in output.")



if __name__ == "__main__":
    unittest.main()
