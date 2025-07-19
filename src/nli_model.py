
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch

class NLIModel:
    def __init__(self, model_name="microsoft/deberta-v2-xxlarge-mnli"):
        # Initialize tokenizer and model
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name)
        self.labels = ["contradiction", "neutral", "entailment"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        # self.labels = ["contradiction", "entailment", "neutral"]

    def nli_inference(self, premise, hypothesis):
        # Tokenize the input
        inputs = self.tokenizer.encode_plus(
            premise, hypothesis, return_tensors='pt', truncation=True
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the predicted label
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        prediction = self.labels[predicted_class]
        
        return prediction

# Example usage
if __name__ == "__main__":
    # # Create an instance of the class
    evidence = 'In 1885, German scientist Hermann Ebbinghaus pioneered the experimental study of memory and is widely acknowledged for his discovery of the forgetting curve, which describes the exponential loss of information over time. Ebbinghaus conducted systematic experiments using nonsense syllables and plotted the decline of memory retention over periods, thereby establishing a quantitative basis for the study of memory and forgetting.'
    A = "One of the first to chart the course of forgetting over time was a psychologist or researcher."
    B = "One of the first to chart the course of forgetting over time was a scientist"