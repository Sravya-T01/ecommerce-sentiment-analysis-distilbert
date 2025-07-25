import os
import gdown
import regex
import torch
import torch.nn as nn
import transformers
from transformers import DistilBertTokenizerFast

def download_model_from_drive():
    model_path = "best_model_state.bin"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1BsFoHw7siSE2iMJZk5psU0ecN_qhigA_"
        gdown.download(url, model_path, quiet=False)


# Define your model class exactly as in training
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.1):
        super(SentimentClassifier, self).__init__()
        self.bert = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        return logits


_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        download_model_from_drive()
        model_path = 'best_model_state.bin'
        _model = SentimentClassifier(n_classes=2, dropout_rate=0.1)  # 2 classes: negative, positive
        _model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        _model.eval()
        _tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    return _model, _tokenizer

def space_emojis(text):
    # Add spaces around emojis using regex to help tokenizer separate them properly
    emoji_pattern = regex.compile(r'(\p{Emoji})')
    spaced_text = emoji_pattern.sub(r' \1 ', text)
    # Clean extra spaces
    spaced_text = ' '.join(spaced_text.split())
    return spaced_text


def predict_sentiment(text):
    text = space_emojis(text)
    model, tokenizer = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs, dim=-1)
    confidence, sentiment_idx = torch.max(probs, dim=1)
    sentiment_map = {0: "Negative", 1: "Positive"}
    sentiment = sentiment_map[sentiment_idx.item()]
    confidence_score = confidence.item() * 100  # convert to percentage
    return sentiment, confidence_score
