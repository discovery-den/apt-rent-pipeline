import logging

import numpy
import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom transformer for applying BERT to text column
class FlairTransformerEmbedding(BaseEstimator, TransformerMixin):
    """
   A transformer class that uses TransformerDocument to embed long text fields.

   Attributes:
       model_name (str): The name of the BERT model to use (default is 'bert-base-uncased').

   Methods:
       fit(X, y=None): Fits the transformer on the input data (no-op for this transformer).
       transform(X): Transforms the input text data into TransformerDocumentEmbeddings.
   """

    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initializes the TransformerDocumentEmbeddings with the specified BERT model and maximum sequence length.

        Args:
            model_name (str): The name of the BERT model to use (default is 'bert-base-uncased').
        """
        self.model_name = model_name
        logger.info(f"Initialized TransformerDocumentEmbeddings with model: {model_name}")

    def fit(self, X, y=None):
        """
        Fits the transformer on the input data (no-op for this transformer).

        Args:
            X (array-like): The input data.
            y (array-like, optional): The target data (not used in this transformer).

        Returns:
            self: The fitted transformer.
        """
        logger.info("Fitting TransformerDocumentEmbeddings (no-op)")
        return self

    def transform(self, X):
        """
        Transforms the input text data into Transformer Document embeddings.

        Args:
            X (array-like): The input text data.

        Returns:
            numpy.ndarray: The Transformer Document embeddings for the input text data.
        """

        model = TransformerDocumentEmbeddings(
            self.model_name, fine_tune=False)

        sentences = [Sentence(text) for text in X]
        embedded = model.embed(sentences)
        embedded = [e.get_embedding().reshape(1, -1) for e in embedded]
        return np.array(torch.cat(embedded).cpu())
