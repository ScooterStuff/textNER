import spacy
from spacy.training import offsets_to_biluo_tags

nlp = spacy.blank("en")  # or load an existing model
text = "I want to play Minecraft"
entities = [(15, 24, "GAME")]  # Example of entity span that might be misaligned

# Create a Doc object
doc = nlp.make_doc(text)

# Check alignment
tags = offsets_to_biluo_tags(doc, entities)
print(tags)


