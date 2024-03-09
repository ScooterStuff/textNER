TRAIN_DATA = [
    ("I want to play Minecraft", {"entities": [(15, 24, "GAME")]}),
    ("Start playing Minecraft with my right hand", {"entities": [(14, 23, "GAME"), (32, 42, "HAND")]}),
    ("place a block down", {"entities": []}),
    ("I love Minecraft", {"entities": []}),
    (",", {"entities": []}),
    ("hold my fist to place a block down", {"entities": [(8, 12, "GESTURE")]}),
    ("Let's play Rocket League", {"entities": [(11, 24, "GAME")]}),
    ("I use the joystick to drive in Horizon", {"entities": [(31, 38, "GAME"), (22, 27, "GESTURE")]}),
    ("FIFA is best played with a controller", {"entities": [(0, 4, "GAME"), (27, 37, "GESTURE")]}),
    ("Playing Tetris requires quick rotations", {"entities": [(8, 14, "GAME")]}),
    ("Cast a spell in Final Fantasy", {"entities": [(16, 29, "GAME")]}),
    ("Score a goal in FIFA using my feet", {"entities": [(16, 20, "GAME")]}),
    ("Build a tower in Minecraft with blocks", {"entities": [(17, 26, "GAME")]}),
    ("Launch the ball in Rocket League with a flip", {"entities": [(19, 32, "GAME")]}),
    ("Rotate pieces in Tetris quickly", {"entities": [(17, 23, "GAME"), (0, 6, "GESTURE")]}),
    ("Drive through the field in Horizon with speed", {"entities": [(27, 34, "GAME"), (0, 5, "GESTURE")]}),
    ("Perform a free kick in FIFA", {"entities": [(23, 27, "GAME"), (10, 19, "GESTURE")]}),
    ("Summon your ally in Final Fantasy", {"entities": [(20, 33, "GAME")]}),
    ("I want to play Fifa with my body", {'entities': [(15, 19, "GAME")]}),
("I want my right hand to control movement of the player", {'entities': [(10, 15, "HAND")]}),
("I want my left hand to control movement of the player", {'entities': [(10, 14, "HAND")]}),
("I want to pass the ball when I have my fist up", {'entities': [(10, 23, "ACTION"), (39,43, "GESTURE")]}),
("I want to sprint when I show three fingers", {'entities': [(10, 16, "ACTION"), (29, 42, "GESTURE")]}),
("I also want to kick the ball when I kick in real life", {'entities': [(15, 28, "ACTION"), (36, 40, "GESTURE")]})
]


#I want to hold button A when I show my three finger
#hold = action
#button A = args (Finer control define smth that doesn't exist)
#three finger = poses
