import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class TextSummarizer:
    def __init__(self):
        # We use 'bart-large-cnn' which is pre-trained specifically for summarization
        self.model_name = "facebook/bart-large-cnn"
        
        # Load the tokenizer (converts text to numbers)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        
        # Load the pre-trained model
        # .to("cpu") ensures it runs on standard laptops without a dedicated GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def summarize(self, text, max_len=130, min_len=30):
        """
        Takes long text and returns a concise summary.
        """
        # Preprocessing: Tokenize the input text
        # 'pt' means return PyTorch tensors
        inputs = self.tokenizer([text], max_length=1024, truncation=True, return_tensors="pt").to(self.device)

        # Inference: Generate summary IDs
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            num_beams=4,               # Beam search for better quality
            max_length=max_len, 
            min_length=min_len, 
            early_stopping=True
        )

        # Postprocessing: Convert IDs back to human-readable text
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary