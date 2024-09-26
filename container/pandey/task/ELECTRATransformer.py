import logging
import numpy as np
from transformers import ElectraTokenizer, ElectraModel
import torch
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ELECTRATransformer(BaseEstimator, TransformerMixin):
    """
    A transformer class that uses ELECTRA to embed German text fields.

    Attributes:
        model_name (str): The name of the ELECTRA model to use (default is 'german-nlp-group/electra-base-german-uncased').
        max_length (int): The maximum sequence length for tokenization (default is 512).

    Methods:
        electra_embed(text): Embeds the given text using the ELECTRA model.
        fit(X, y=None): Fits the transformer on the input data (no-op for this transformer).
        transform(X): Transforms the input text data into ELECTRA embeddings.
    """

    def __init__(self, model_name: str = 'german-nlp-group/electra-base-german-uncased', max_length: int = 512):
        """
        Initializes the ELECTRATransformer with the specified ELECTRA model and maximum sequence length.

        Args:
            model_name (str): The name of the ELECTRA model to use (default is 'german-nlp-group/electra-base-german-uncased').
            max_length (int): The maximum sequence length for tokenization (default is 512).
        """
        self.model_name = model_name
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraModel.from_pretrained(model_name)
        self.max_length = max_length
        logger.info(f"Initialized ELECTRATransformer with model: {model_name} and max_length: {max_length}")

    def electra_embed(self, text):
        """
        Embeds the given text using the ELECTRA model.

        Args:
            text (str): The text to be embedded.

        Returns:
            numpy.ndarray: The ELECTRA embeddings for the text.
        """
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the embeddings from the [CLS] token (first token) as the representation
        embeddings = outputs.last_hidden_state[:, 0].numpy()  # Shape: (1, hidden_size)
        logger.debug(f"Embedded text: {text}")
        return embeddings

    def fit(self, X, y=None):
        """
        Fits the transformer on the input data (no-op for this transformer).

        Args:
            X (array-like): The input data.
            y (array-like, optional): The target data (not used in this transformer).

        Returns:
            self: The fitted transformer.
        """
        logger.info("Fitting ELECTRATransformer (no-op)")
        return self

    def transform(self, X):
        """
        Transforms the input text data into ELECTRA embeddings.

        Args:
            X (array-like): The input text data.

        Returns:
            numpy.ndarray: The ELECTRA embeddings for the input text data.
        """
        logger.info("Transforming text data into ELECTRA embeddings")
        embeddings = [self.electra_embed(text) for text in X]
        embeddings = np.concatenate(embeddings)
        logger.info(f"Transformed {len(X)} texts into ELECTRA embeddings")
        return embeddings
