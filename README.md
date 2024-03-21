# MotionInput Configuration Generator
The MotionInput Configuration Generator is an innovative NLP/NER system designed to bridge the gap between natural language instructions and motion-based game controls. Utilizing cutting-edge language processing technologies, this project translates user-provided speech inputs into JSON configurations. These configurations enable the MotionInput program, which leverages OpenCV for motion detection, to dynamically map specific body movements or gestures to in-game actions, allowing for a customizable and immersive gaming experience.

## Table of Contents
1. [Overview](#overview)
2. [Technologies](#technologies)
3. [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Usage](#usage)
    - [Example Input](#example-input)
    - [Example Output](#example-output)
5. [MotionInput NER Model Training](#motioninput-ner-model-training)
    - [Overview](#overview-1)
    - [Training the Model](#training-the-model)
    - [Customization](#customization)
6. [Testing](#testing)
    - [Prerequisites](#prerequisites-1)
    - [Running the Tests](#running-the-tests)
    - [Writing and Adding New Tests](#writing-and-adding-new-tests)
    - [Continuous Integration](#continuous-integration)
7. [Maintenance Guide](#maintenance-guide)
    - [Updating Entity Matchers](#introduction)
    - [Extending Semantic Similarity Matching](#component-breakdown)
    - [Updating Games Controls Mapping and Possible Gestures/Poses](#maintenance-guide)

## Overview
This project integrates speech recognition, Named Entity Recognition (NER), and semantic similarity matching to interpret the user's game preferences and desired control schemes. The system supports a variety of games and gestures, understanding instructions such as "I want to play Minecraft with my left hand, using two fingers to place a block," and translating these into actionable configurations.

### Technologies
* SpaCy: For Named Entity Recognition (NER), identifying key elements in the speech such as game names, body parts, and actions.
* Sentence Transformers: Employed for semantic similarity matching, ensuring the user's described actions are accurately mapped to pre-defined in-game actions.
* OpenCV: For motion detection, working in tandem with the generated configuration to recognize and interpret player gestures.

## Setup
### Prerequisites
* Python 3.x
* SpaCy
* Sentence Transformers
* SciPy

Ensure you have Python 3.11+ installed on your system. Then, install the required Python packages using pip:
```
pip install -r requirements.txt
```

### Installation
Clone this repository to your local machine.
git clone https://github.com/ScooterStuff/textNER.git
Navigate to the project directory.
Download and install the SpaCy language model.
python -m spacy download en_core_web_sm

Ensure you have the necessary models and configurations placed within the project directory as per the given structure:
./fine-tuned-model: Directory for the Sentence Transformer model.
./ner_model: Directory for the SpaCy NER model.

## Usage

To use the MotionInput Configuration Generator, follow these steps:
Prepare your speech input for the game and controls configuration.
Run the main script to generate the JSON configuration.
python main.py
The script will output a JSON file (prediction_output.json) in the project directory, which contains the mappings for game controls based on your input.

### Example Input
"I want to play Minecraft with my left hand use two fingers to place down a block."

### Example Output
```
{
    "mode": "Minecraft",
    "orientation": "left",
    "landmark": "hand",
    "poses": [
        {
            "files": "two_fingers",
            "action": {
                "tmpt": "place down",
                "class": "right_click",
                "method": "hold",
                "args": ["right"]
            }
        }
    ]
}
```


# MotionInput NER Model Training

This repository contains the training script for a Named Entity Recognition (NER) model designed for the MotionInput Configuration Generator. The script uses SpaCy to train an NER model capable of identifying games, gestures, and poses from natural language inputs, facilitating the creation of customized control configurations for motion-based game interactions.

## Overview
The training script utilizes a custom dataset and SpaCy's powerful NLP capabilities to recognize and label specific entities within user inputs. These entities include:

* Games: Recognizes game titles like Minecraft, Tetris, and others.
* Orientation: Choose left or right part of landmark
* Landmark: Where the main body part of controls
* Gestures: Identifies specific gestures mentioned in the input, such as "hadouken" or "front kick".
* Poses: Detects pose descriptions, like "thumb up" or "index pinch".
* The trained model can then be integrated with the MotionInput Configuration Generator to interpret user-defined control schemes.

## Training the Model
The script train_ner.py is used to train the NER model. Before running the script, ensure you have prepared your training data in the format expected by SpaCy. The script expects a file named train_data.py containing the variable TRAIN_DATA, which holds the training examples.

### Structure of TRAIN_DATA:
TRAIN_DATA should be a list of tuples, where each tuple represents a training example. Each tuple consists of the text and a dictionary with the key "entities" pointing to a list of entity annotations. Each entity annotation is a tuple of (start_index, end_index, label).

### Running the Training Script
To train your model, simply run:
```
python train_ner.py
```
The script will train the NER model using the provided data and save it to the ./ner_model directory.

## Customization
You can customize the training process by modifying the train_ner function in the script. Parameters such as the model directory (model_dir), the training data (new_data), the number of training iterations (n_iter) can be adjusted to suit your needs and the dropout rate (drop).


# Testing
This project includes a suite of tests to ensure the functionality, integration, and performance of the MotionInput Configuration Generator and its associated NER model. Below you will find instructions on how to run these tests, which are divided into unit tests and integration tests.

## Prerequisites
Before running the tests, ensure you have the following installed:

* Python 3.x
* The required libraries for the project (SpaCy, etc.)
* unittest (part of the standard library in Python)

## Running the Tests
The tests are organized into three files:

* test_predict.py contains unit tests for the prediction functionality, focusing on individual components of the system.
* test_integration.py includes integration tests that assess the system as a whole, ensuring that all components work together correctly.
* test_train_ner.py features tests for the training functionality, verifying the NER model training process.

### Unit Tests
To run the unit tests, navigate to the project directory in your terminal or command prompt and run:
```
python -m unittest test_predict.py
```
This command will execute the tests defined in test_predict.py, checking the correctness of the prediction logic.

### Integration Tests
Similarly, to run the integration tests, use the following command:
```
python -m unittest test_integration.py
```
The integration tests in test_integration.py evaluate the entire pipeline from processing input sentences to generating the correct JSON configuration output.

### NER Model Training Tests
To test the NER model training functionality, run:
```
python -m unittest test_train_ner.py
```

These tests ensure that the custom entity matchers and the training process function as intended.

## Writing and Adding New Tests
When adding new tests or functionality, please ensure you also add corresponding unit or integration tests. This helps maintain the reliability and robustness of the system over time.

1. For new functionality: Add unit tests in the relevant file (or create a new one if necessary) that cover the expected behavior, edge cases, and potential errors.
2. For bug fixes: Write a test that reproduces the bug (which should fail before the fix) and passes after the fix is applied.
Ensure all tests pass before submitting a pull request or making significant changes to the codebase.

## Continuous Integration
Consider setting up continuous integration (CI) to automatically run these tests on every commit or pull request. This will help catch issues early and improve the quality of contributions.

# Maintainance Breakdown

### Introduction
This system translates natural language descriptions of desired game control schemes into JSON configurations. These configurations dictate how physical gestures (identified via OpenCV) map to in-game actions. It hinges on Named Entity Recognition (NER) for identifying games, gestures, and actions, and employs semantic similarity matching to align described actions with predefined in-game commands.

### Component Breakdown
1. Loading Models
SentenceTransformer Model: Used for semantic similarity matching. It's loaded from MODEL_PATH and is essential for comparing the semantic similarity between the user's described actions and the available actions within a specific game.
SpaCy Model: Loaded from NER_MODEL_PATH, this model has been extended with custom matchers (game, gesture, pose) to recognize specific entities within the user's input.
2. Custom Entity Matchers
Game Entity Matcher: Identifies game titles within the user's input. It's crucial for determining which game's control scheme is being configured.
Gesture and Pose Matchers: These identify specific gestures or poses mentioned by the user. Gestures might include actions like "hadouken" or "punch," while poses could be specific hand or finger positions.
3. Semantic Similarity Matching (similarties_match)
This function computes the semantic similarity between the user's described action and a set of possible in-game actions using embeddings generated by the SentenceTransformer model. It ensures that even if the user's description doesn't exactly match the predefined action terms, the closest match can still be identified and used.

4. Motion to Action Mapping (motion_to_action_mapping)
After identifying the closest semantic match, this mapping translates it into a specific in-game action, like "jump" or "crouch." This mapping depends on a predefined list of actions supported per game.

5. Processing Input (predict_without_commas)
This function orchestrates the processing of user input:

Splits the input into manageable sentences.
Uses the NER model to identify and categorize entities (games, gestures, poses, etc.).
Applies similarity matching and motion-to-action mapping to generate a detailed action configuration.
6. Generating Configuration Output (predict_to_json)
Finally, this function compiles the processed data into a JSON configuration file. This file serves as the output, detailing how each gesture or pose is mapped to a specific in-game action.

## Maintenance Guide
### Updating Entity Matchers
Matcher is a tool in Spacy that lets you search for sequences of tokens that match specific patterns. It's part of SpaCy's powerful processing pipeline and is used for identifying and extracting pieces of text based on criteria you define, without the need for regular expressions. This is particularly useful for Named Entity Recognition (NER), where you might want to identify specific terms or phrases in your text that match certain patterns.
To expand the system's capabilities for recognizing new games, gestures, or poses, follow these steps:

1. **Edit `game_entity_matcher`** to add more game:
   - **Games Available**: Extend the `patterns` list to include new games. This list defines what clasified as GAME entities, so without it existing in the train_data set if the sentence pattern is similar to one that recognizes game it will then recognize it as a game.
   - **Wording**: If the word have space in the middle e.g. Rocket League you need to do `[{"LOWER": "rocket"}, {"LOWER": "league"}]`, which then SpaCy will recognizes as one entity, also note that this can be use with regex e.g. `[{"LOWER": "final"}, {"LOWER": "fantasy"}, {"TEXT": {"REGEX": "^(X|V|I)+$"}}],`
2. **Application to Other Matcher**:
    - **gesture_entity_matcher**
    - **pose_entity_matcher**

### Extending Semantic Similarity Matching
To improve or extend the similarity matching:
1. **Consider retraining the SentenceTransformer model in `sim_nlp.py`**  on a more specific dataset if the current model struggles with accurately matching game-specific terminology.
  - **Example**
  ```python
  data = [
        {"sentence1": "move the ball", "sentence2": "pass", "similarity": 0.9},
        {"sentence1": "move the ball", "sentence2": "walk", "similarity": 0.1}]
```

2. **Adjust the similarity threshold** in similarties_match if necessary to fine-tune the balance between matching accuracy and leniency.

### Updating Games Controls Mapping and Possible Gestures/Poses
To expand the system's capabilities for recognizing new games, gestures, or poses. Which may later be added to MotionInput, follow these steps:

1. **Edit `game_controls.py`** to adjust game actions and key mappings:
   - **Games Actions**: Extend the `games_actions` dictionary to include new games and their associated actions. This dictionary defines what actions (like jumping, running, placing items) are available for each game.
   - **Key Mappings**: Update the `game_key_mappings` dictionary to map the newly added actions to specific keyboard inputs or controller buttons.
   Example of adding a new game and its actions:
   ```python
   games_actions["NewGame"] = ["action1", "action2", "action3"]
   game_key_mappings["NewGame"] = {
       "action1": "key1",
       "action2": "key2",
       "action3": "key3"
   }
2. **Edit `available_gesture_and_pose.py`** to adjust available poses and gesture:
   - **Possible Gestures/Poses**: Extend the `available_gestures` or `available_poses` dictionary to include new games and their associated actions. This dictionary defines what actions (like jumping, running, placing items) are available for each game.
   
   Example of adding a new game and its actions:
   ```python
   available_poses = ['fist', 'fist2',...]
   available_gestures = ['bow_arrow', 'fighting_stance',...]

## General Tips
* Regularly update the underlying models (SpaCy, Sentence Transformers) to benefit from improvements in NLP technology.
* Keep the training data for both NER and semantic similarity models current with new game releases and popular terminology to ensure the system remains relevant and effective.

This guide aims to equip maintainers with a comprehensive understanding of the system's architecture and functions, facilitating effective management, troubleshooting, and enhancement of the system.

## Indepth Explaination
* https://students.cs.ucl.ac.uk/2023/group21/
* Credit: Tatsan Kantasit
