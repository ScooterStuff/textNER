gesture_patterns = [
    [{"LOWER": "bow"}, {"LOWER": "arrow"}],
    [{"LOWER": "fighting"}, {"LOWER": "stance"}],
    [{"LOWER": "front"}, {"LOWER": "kick"}],
    [{"LOWER": "hadouken"}],
    [{"LOWER": "helicopter"}],
    [{"LOWER": "index"}, {"LOWER": "pinch"}],
    [{"LOWER": "kick"}],
    [{"LOWER": "left"}, {"LOWER": "hook"}],
    [{"LOWER": "left"}, {"LOWER": "kick"}],
    [{"LOWER": "left"}, {"LOWER": "punch"}],
    [{"LOWER": "mine"}],
    [{"LOWER": "punch"}],
    [{"LOWER": "push"}, {"LOWER": "back"}],
    [{"LOWER": "right"}, {"LOWER": "clockwise"}, {"LOWER": "circle"}],
    [{"LOWER": "right"}, {"LOWER": "hook"}],
    [{"LOWER": "right"}, {"LOWER": "kick"}],
    [{"LOWER": "right"}, {"LOWER": "punch"}],
    [{"LOWER": "uppercut"}],
    [{"LOWER": "walk"}, {"LOWER": "left"}],
    [{"LOWER": "walk"}, {"LOWER": "right"}],
]

# Creating a list where each element's name is derived from the gesture patterns
gesture_array = ['_'.join([item['LOWER'] for item in pattern]) for pattern in gesture_patterns]

print(gesture_array)