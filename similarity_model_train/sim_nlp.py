from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader  # Corrected import
import pandas as pd

class ModelFineTuner:
    def __init__(self, base_model='all-MiniLM-L6-v2', output_path='./fine-tuned-model'):
        self.model = SentenceTransformer(base_model)
        self.output_path = output_path

    def prepare_data(self, data):
        """
        Convert a list of dictionaries to a DataLoader with InputExample objects.
        """
        df = pd.DataFrame(data)
        examples = [InputExample(texts=[row['sentence1'], row['sentence2']], label=row['similarity']) for index, row in df.iterrows()]
        dataloader = DataLoader(examples, shuffle=True, batch_size=2)
        return dataloader

    def fine_tune(self, data, epochs=4, warmup_steps=100):
        """
        Fine-tune the model on the provided dataset.
        """
        train_dataloader = self.prepare_data(data)
        train_loss = losses.CosineSimilarityLoss(model=self.model)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       epochs=epochs,
                       warmup_steps=warmup_steps,
                       output_path=self.output_path)

    def load_fine_tuned_model(self):
        """
        Load the fine-tuned model from the output path.
        """
        self.model = SentenceTransformer(self.output_path)

# Example usage:

if __name__ == "__main__":
    fine_tuner = ModelFineTuner()

    # Simulated dataset
    data = [
        {"sentence1": "move the ball", "sentence2": "pass", "similarity": 0.9},
        {"sentence1": "move the ball", "sentence2": "walk", "similarity": 0.1},
        {"sentence1": "move the ball", "sentence2": "punt", "similarity": 0.8},
        {"sentence1": "move the ball", "sentence2": "run", "similarity": 0.2},
    ]

    fine_tuner.fine_tune(data=data)
    fine_tuner.load_fine_tuned_model()

    # Now, you can use fine_tuner.model as before to generate embeddings and calculate similarities
