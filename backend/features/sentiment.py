from transformers import pipeline
import torch
from backend.core.logging import logger
from typing import List, Dict, Union
import numpy as np

class SentimentAnalyzer:
    """
    Analyzes financial text sentiment using FinBERT.
    """
    def __init__(self):
        # Check for GPU
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Initializing FinBERT on device: {'GPU' if device == 0 else 'CPU'}")
        
        try:
            self.pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device)
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.pipe = None

    def analyze(self, text: Union[str, List[str]]) -> float:
        """
        Analyze sentiment of text(s).
        Returns a score between -1.0 (Negative) and 1.0 (Positive).
        """
        if not self.pipe:
            logger.warning("FinBERT not initialized, returning neutral score.")
            return 0.0

        if isinstance(text, str):
            text = [text]

        # FinBERT limits
        # We should truncate or chunk if too long, but for headlines, it's fine.
        try:
            results = self.pipe(text, padding=True, truncation=True, max_length=512)
            
            # Convert labels to score
            # positive: 1.0, neutral: 0.0, negative: -1.0
            # weight by score confidence
            
            total_score = 0
            for res in results:
                label = res['label']
                score = res['score']
                
                if label == 'positive':
                    total_score += score
                elif label == 'negative':
                    total_score -= score
                # neutral contributes 0
            
            # Average score
            return total_score / len(text)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0

sentiment_analyzer = SentimentAnalyzer()
