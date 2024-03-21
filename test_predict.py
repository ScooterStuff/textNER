import unittest
from unittest.mock import patch, mock_open
from predict_org import similarties_match, motion_to_action_mapping, predict_to_json, initialize_output_structure, predict_without_comma
from unittest.mock import MagicMock

class TestNLPNERModel(unittest.TestCase):

    def test_similarities_match(self):
        target_phrase = "mine"
        possible_phrases = ["place", "mine", "break"]
        # Assuming 'mine' has the highest similarity score.
        expected = "mine"
        result = similarties_match(target_phrase, possible_phrases)
        self.assertEqual(result, expected)

    def test_motion_to_action_mapping(self):
        motion = "jump"
        game = "Minecraft"
        expected = "jump"  # Assuming 'jump' is the closest motion for Minecraft.
        result = motion_to_action_mapping(motion, game)
        self.assertEqual(result, expected)

    @patch('predict_org.json.dump')
    @patch('predict_org.open', new_callable=mock_open, create=True)
    def test_predict_to_json(self, mock_open, mock_json_dump):
        sentences = "I want to play Minecraft with my left hand, jump using two fingers."
        output_file = "test_output.json"
        predict_to_json(sentences, output_file)
        mock_open.assert_called_with(output_file, 'w', encoding='utf-8')
        # This checks if json.dump was called, indicating the output was written to file.
        self.assertTrue(mock_json_dump.called)


    def test_initialize_output_structure(self):
        expected = {"mode": "", "orientation": "", "landmark": "", "poses": [], "gestures": []}
        result = initialize_output_structure()
        self.assertEqual(result, expected)

    @patch('predict_org.nlp')
    @patch('predict_org.sentence_model.encode')
    def test_predict(self, mock_encode, mock_nlp):
        # Prepare the sentence embeddings mock
        mock_encode.return_value = ["some", "encoded", "values"]

        # Create mock entities with label_ and text attributes
        mock_game_ent = MagicMock()
        mock_game_ent.label_ = 'GAME'
        mock_game_ent.text = 'Tetris'

        mock_action_ent = MagicMock()
        mock_action_ent.label_ = 'ACTION-O'
        mock_action_ent.text = 'Rotate item'

        # Create a mock doc with an ents attribute returning a list of mock entities
        mock_doc = MagicMock()
        mock_doc.ents = [mock_game_ent, mock_action_ent]

        # Set the nlp mock's return value to our mock doc
        mock_nlp.return_value = mock_doc

        # Initialize the output structure
        output_data = initialize_output_structure()

        # Call predict with the mocks
        split_sentences = ["I want to play Tetris", "Rotate item using hadouken"]
        predict_without_comma(split_sentences, output_data)

        # Assert based on the expected changes to output_data
        # Example assertion: check if 'mode' in output_data has been set to 'Tetris'
        self.assertEqual(output_data["mode"], 'Tetris')
        # Additional assertions can be made based on the specific logic of your `predict` function
        # For example, checking if certain actions, poses, or gestures have been added to output_data

    def test_motion_to_action_mapping_returns_none_for_unknown_game(self):
        motion = "fly"
        game = "UnknownGame"
        expected = "Game not found"
        result = motion_to_action_mapping(motion, game)
        self.assertEqual(result, expected)

    @patch('predict_org.json.dump')
    @patch('predict_org.open', mock_open(), create=True)
    def test_predict_to_json_creates_correct_file_structure(self, mock_json_dump):
        sentences = "I want to play Minecraft; Jump using two fingers."
        output_file = "test_output_structure.json"
        predict_to_json(sentences, output_file)
        # Assuming mock_json_dump is called with the correct structure, you can inspect the call to ensure it matches expectations
        args, kwargs = mock_json_dump.call_args
        output_data = args[0]  # json.dump is called with the data as the first argument
        expected_keys = ["mode", "orientation", "landmark", "poses", "gestures"]
        self.assertTrue(all(key in output_data for key in expected_keys))
        # Further checks can verify the structure of poses, gestures, etc.

    def test_similarities_match_with_empty_possible_phrases(self):
        target_phrase = "mine"
        possible_phrases = []
        expected = "none"  # Assuming we return "none" when there are no possible phrases to compare with
        result = similarties_match(target_phrase, possible_phrases)
        self.assertEqual(result, expected)

    def test_motion_to_action_mapping_with_invalid_game_type(self):
        motion = "jump"
        game = {"name": "Minecraft"}  # Non-string game
        with self.assertRaises(TypeError):
            motion_to_action_mapping(motion, game)

    def test_similarities_match_with_similarity_scores(self):
        target_phrase = "sprint"
        possible_phrases = ["walk", "run", "jump"]  # Assuming low similarity with "sprint"
        expected = "run"
        result = similarties_match(target_phrase, possible_phrases)
        self.assertEqual(result, expected, "Expected 'none' when similarities are low")


    @patch('predict_org.json.dump')
    @patch('predict_org.open', mock_open(), create=True)
    def test_predict_to_json_with_empty_sentences(self, mock_json_dump):
        sentences = ""
        output_file = "empty_test_output.json"
        predict_to_json(sentences, output_file)
        args, kwargs = mock_json_dump.call_args
        output_data = args[0]
        expected_keys = ["mode", "orientation", "landmark", "poses", "gestures"]
        self.assertTrue(all(key in output_data for key in expected_keys))
        # Ensure that the output for empty sentences does not populate any unnecessary data
        self.assertTrue(all(not output_data[key] for key in expected_keys))
    

    @patch('predict_org.predict_without_comma')
    @patch('predict_org.json.dump')
    @patch('predict_org.open', new_callable=mock_open, create=True)
    def test_predict_to_json_content_verification(self, mock_file_open, mock_json_dump, mock_predict_without_comma):
        sentences = "I want to play Minecraft; Jump using two fingers."
        output_file = "content_verification_output.json"
        mock_output_data = {
            "mode": "Minecraft",
            "orientation": "",
            "landmark": "",
            "poses": [],
            "gestures": [{"files": "two_fingers", "action": {"tmpt": "jump", "class": "jump", "method": "click", "args": ["space"]}}]
        }
        mock_predict_without_comma.side_effect = lambda s, o: o.update(mock_output_data)
        
        predict_to_json(sentences, output_file)
        
        args, kwargs = mock_json_dump.call_args
        output_data = args[0]
        self.assertEqual(output_data, mock_output_data, "The content of the output file did not match the expected output")
    
    def test_similarities_match_with_empty_inputs(self):
        # Test with empty list of possible phrases
        expected = "none"
        target_phrase = "mine"
        possible_phrases = []
        result = similarties_match(target_phrase, possible_phrases)
        self.assertEqual(result, expected, "Expected 'none' for an empty list of possible phrases.")
    
    @patch('predict_org.json.dump')
    @patch('predict_org.open', mock_open(), create=True)
    def test_predict_to_json_with_empty_sentences(self, mock_json_dump):
        sentences = ""
        output_file = "empty_sentences_output.json"
        predict_to_json(sentences, output_file)
        args, kwargs = mock_json_dump.call_args
        output_data = args[0]
        expected_keys = ["mode", "orientation", "landmark", "poses", "gestures"]
        self.assertTrue(all(key in output_data for key in expected_keys), "All expected keys should be present in the output.")
        self.assertTrue(all(not output_data[key] for key in expected_keys), "Output for empty sentences should not populate any data.")



if __name__ == '__main__':
    unittest.main()
