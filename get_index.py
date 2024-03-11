import spacy

nlp = spacy.blank("en")  # Assuming a blank English model
text = "I use the joystick to drive in Horizon"
ls = [
    "I want my right hand to control movement of the player",
    "I want to pass the ball when I have my fist up",
    "I want to sprint when I show three fingers",
    "I also want to kick the ball when I kick in real life",
]
for l in ls:
    print(l)
    doc = nlp(l)

    # Print out tokenization results
    print([(token.text, token.idx, token.idx+len(token)) for token in doc])


text = "During my gaming session, I prefer playing Rocket League with intuitive body movements; I use my left hand for navigation, clenching a fist to boost speed, pointing with two fingers to change direction, and making a kicking motion to shoot the ball towards the goal."

# Splitting the text into an array of sentences based on ",", ".", ";", and "and"
split_sentences = [sentence.strip() for sentence in text.replace('and', ',').replace(';', ',').replace('.', ',').split(',') if sentence]

print(split_sentences)