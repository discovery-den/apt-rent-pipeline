import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.base import BaseEstimator, TransformerMixin


class BERTTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for transforming text columns using a BERT model.

    This class tokenizes text and generates BERT embeddings for text data,
    returning the [CLS] token embeddings in a 2D array format for each input sample.

    Attributes:
        model_name (str): The name of the pre-trained BERT model to use.
        max_length (int): The maximum length for the tokenized sequences.
    """

    def __init__(self, model_name='bert-base-german-cased', max_length=128):
        """
        Initializes the BERTTransformer with the specified BERT model.

        Args:
            model_name (str): The name of the BERT model to use (default is 'bert-base-uncased').
            max_length (int): The maximum token length for each input sequence.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def fit(self, X, y=None):
        """
        No-op fit method for compatibility with scikit-learn pipelines.

        Args:
            X (array-like): The input data.
            y (array-like, optional): The target data (ignored).

        Returns:
            self: Fitted transformer instance.
        """
        return self

    def transform(self, X):
        """
        Transforms the input text data into BERT [CLS] token embeddings.

        Args:
            X (array-like): The input text data.

        Returns:
            np.ndarray: 2D array where each row corresponds to a [CLS] token embedding for each input text.
        """
        # Tokenize and generate embeddings for all rows
        embeddings = [self._get_cls_embedding(text) for text in X]

        # Ensure all embeddings have the correct dimensions (reshape if necessary)
        embeddings = np.concatenate(embeddings)
        return embeddings

    def _get_cls_embedding(self, text):
        """
        Tokenizes the input text and generates the [CLS] token embedding from the BERT model.

        Args:
            text (str): The input text string.

        Returns:
            np.ndarray: The [CLS] token embedding.
        """
        # Handle empty or missing text by using a default token
        if not text or not isinstance(text, str):
            text = "[CLS]"

        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)

        # Pass the tokenized input to the model to obtain embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the [CLS] token embedding (the first token)
        cls_embedding = outputs.last_hidden_state[:, 0].numpy()  # Shape: (1, hidden_size)

        return cls_embedding
