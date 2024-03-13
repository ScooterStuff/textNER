TRAIN_DATA = [
    ("I want to play Minecraft", {"entities": [(15, 24, "GAME")]}),
    ("Start playing Minecraft with my right hand", {"entities": [(14, 23, "GAME"), (32, 37, "ORI"), (38, 42, "LANDMARK")]}),
    ("place a block down", {"entities": []}),
    ("I love Minecraft", {"entities": []}),
    (",", {"entities": []}),
    ("hold my fist to place a block down", {"entities": [(8, 12, "POSES"), (16, 34, "GESTURE")]}),
    ("I want to pass the ball when I have my fist up", {'entities': [(10, 23, "ACTION-O"), (39,43, "POSES")]}),
    ("I want to sprint when I show three fingers", {'entities': [(10, 16, "ACTION-O"), (29, 42, "GESTURE")]}),
    ("I also want to kick the ball when I kick in real life", {'entities': [(15, 28, "ACTION-O"), (36, 40, "GESTURE")]}),
    ("I want to activate Mario with my left hand", {'entities': [(19, 24, "GAME"), (33,37, "ORI"), (38, 42, "LANDMARK")]}),
    ("I want to cast a spell with an index pinch", {"entities": [(10, 22, "GESTURE"),(31, 42, "GESTURE")]}),
    ("Block attacks with a thumb index pinch", {"entities": [(0,13, "ACTION-O"),(21, 38, "POSES")]}),
    ("Use the index pinch to interact with objects", {"entities": [(8, 19, "POSES"),(23, 44, "ACTION-O")]}),
    ("To crouch just do a thumb down", {"entities": [(3,9, "ACTION-O"),(20, 30, "POSES")]}),
    ("Rotate item with a three fingers pinch", {"entities": [(0,11, "ACTION-O"),(19,38, "GESTURE")]}),
]


#I want to hold button A when I show my three finger
#hold = action
#button A = args (Finer control define smth that doesn't exist)
#three finger = poses


# Not Indepth Mode (This would assumes most of thing map to most of things)

# Mario - GAME
# left - ORITEINTATION
# hand - LANDMARK
# fist - POSES - FILE
# double pinch - GESTURE - FILE
# jump - ACTION-O
# press - ACTION-C
# B/Shift - ACTION-A