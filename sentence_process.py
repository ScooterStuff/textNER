import time
start = time.time()
from deepmultilingualpunctuation import PunctuationModel
model = PunctuationModel()
text = "I want to play FIFA with my body I want my right hand to control movement of the player I want to put my fist up to pass the ball I want to sprint when I show three finger I also want to kick the ball when I Kick In Real Life"
result = model.restore_punctuation(text)
print(result)
end = time.time()
print(end - start)