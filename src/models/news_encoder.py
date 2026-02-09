"""
News encoder using FinBERT for financial text embeddings and sentiment analysis.
"""

from __future__ import annotations

from typing import Optional, List, Union
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel


class FinBERTEncoder:
    """
    Encoder for financial news using FinBERT.
    
    FinBERT is a BERT model specifically trained on financial texts.
    It can be used for:
    1. Sentiment analysis (positive/negative/neutral)
    2. Text embeddings (for neural network input)
    
    Model: ProsusAI/finbert
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        use_sentiment: bool = True,
    ):
        """
        Initialize FinBERT encoder.
        
        Args:
            model_name: Hugging Face model name (default: ProsusAI/finbert)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            use_sentiment: Whether to use sentiment classification head
        """
        self.model_name = model_name
        self.use_sentiment = use_sentiment
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading FinBERT model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        if use_sentiment:
            # Load with classification head for sentiment
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_labels = ["positive", "negative", "neutral"]
        else:
            # Load base model for embeddings only
            self.model = AutoModel.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ FinBERT loaded successfully")
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        return_sentiment: bool = False,
        max_length: int = 512,
    ) -> Union[np.ndarray, tuple[np.ndarray, Optional[dict]]]:
        """
        Encode text(s) into embeddings.
        
        Args:
            text: Single text string or list of texts
            return_sentiment: Whether to also return sentiment scores
            max_length: Maximum sequence length
        
        Returns:
            If return_sentiment=False: numpy array of embeddings (shape: [batch_size, hidden_size])
            If return_sentiment=True: tuple of (embeddings, sentiment_dict)
        """
        # Handle single text
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            if self.use_sentiment:
                # Use BERT base model for embeddings
                outputs = self.model.bert(**encoded)
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Get sentiment if requested
        sentiment_dict = None
        if return_sentiment and self.use_sentiment:
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            
            sentiment_dict = {
                'labels': self.sentiment_labels,
                'probabilities': probs,
                'predicted': [self.sentiment_labels[np.argmax(p)] for p in probs],
            }
        
        # Return single embedding if single input
        if single_input:
            embeddings = embeddings[0]
            if sentiment_dict:
                sentiment_dict = {
                    'label': sentiment_dict['predicted'][0],
                    'probabilities': {
                        label: prob
                        for label, prob in zip(
                            sentiment_dict['labels'],
                            sentiment_dict['probabilities'][0]
                        )
                    },
                }
        
        if return_sentiment:
            return embeddings, sentiment_dict
        return embeddings
    
    def get_sentiment(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
    ) -> Union[dict, List[dict]]:
        """
        Get sentiment analysis for text(s).
        
        Args:
            text: Single text string or list of texts
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with sentiment label and probabilities
        """
        if not self.use_sentiment:
            raise ValueError("Model was initialized without sentiment head")
        
        _, sentiment = self.encode_text(text, return_sentiment=True, max_length=max_length)
        return sentiment
    
    def encode_news_batch(
        self,
        news_texts: List[str],
        batch_size: int = 32,
        return_sentiment: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, List[dict]]]:
        """
        Encode a batch of news texts efficiently.
        
        Args:
            news_texts: List of news text strings
            batch_size: Batch size for processing
            return_sentiment: Whether to return sentiment scores
        
        Returns:
            Embeddings array and optionally sentiment scores
        """
        all_embeddings = []
        all_sentiments = []
        
        for i in range(0, len(news_texts), batch_size):
            batch = news_texts[i:i + batch_size]
            
            if return_sentiment:
                embeddings, sentiments = self.encode_text(
                    batch,
                    return_sentiment=True
                )
                all_sentiments.extend(sentiments if isinstance(sentiments, list) else [sentiments])
            else:
                embeddings = self.encode_text(batch, return_sentiment=False)
            
            all_embeddings.append(embeddings)
        
        # Concatenate embeddings
        embeddings_array = np.vstack(all_embeddings)
        
        if return_sentiment:
            return embeddings_array, all_sentiments
        return embeddings_array


__all__ = ['FinBERTEncoder']
