import logging
import time
from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from BERTTransformer import BERTTransformer

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
GREEN = '\033[92m'
ENDC = '\033[0m'


def preprocess_numeric_columns() -> Pipeline:
    """
    Creates a pipeline for preprocessing numeric columns.

    Returns:
        Pipeline: A scikit-learn pipeline for preprocessing numeric columns.
    """
    # Log the start of the preprocessing process
    logger.info("Starting preprocessing of numeric columns.")

    # Record the start time
    start_time = time.time()

    # Create a pipeline for imputation and scaling
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Log the end of the preprocessing process and the time taken
    end_time = time.time()
    logger.info(f"{GREEN}Preprocessing completed. Time taken: {end_time - start_time:.4f} seconds{ENDC}")

    return numeric_pipeline


def preprocess_categorical_columns() -> Pipeline:
    """
    Creates a pipeline for preprocessing categorical columns.

    Returns:
        Pipeline: A scikit-learn pipeline for preprocessing categorical columns.
    """
    # Log the start of the preprocessing process
    logger.info("Starting preprocessing of categorical columns.")

    # Record the start time
    start_time = time.time()

    # Create a pipeline for one-hot encoding and feature selection
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Log the end of the preprocessing process and the time taken
    end_time = time.time()
    logger.info(f"{GREEN}Preprocessing completed. Time taken: {end_time - start_time:.4f} seconds{ENDC}")

    return categorical_pipeline


def preprocess_columns_with_long_text(category_columns: list = None,
                                      long_text_columns: list = None) -> ColumnTransformer:
    """
    Creates a ColumnTransformer to apply separate pipelines for numerical, categorical and long text fields columns.

    Returns:
        ColumnTransformer: A ColumnTransformer object that applies the given numerical, categorical and BERT
                            Transformation pipelines to the respective columns.
    """

    logger.info(f"Starting columns transformation with long text fields. {long_text_columns}")

    # Record the start time
    start_time = time.time()
    preprocessor = ColumnTransformer(transformers=[
        ('num', preprocess_numeric_columns(), selector(dtype_include=np.number)),  # Apply numerical pipeline
        ('cat', preprocess_categorical_columns(), category_columns),  # Apply categorical pipeline
        ('trans_description', BERTTransformer(), 'description'),  # Apply Custom BERTTransformer on long text field
        ('trans_facilities', BERTTransformer(), 'facilities'),
    ])

    # Log the end of the transformation process and the time taken
    end_time = time.time()
    logger.info(
        f"{GREEN}Transformation completed including custom BERT Transformation. Time taken: {end_time - start_time:.4f}"
        f" seconds{ENDC}")

    return preprocessor


def preprocess_columns() -> ColumnTransformer:
    """
    Creates a ColumnTransformer to apply separate pipelines for numerical and categorical columns.

    Returns:
        ColumnTransformer: A ColumnTransformer object that applies the given numerical and categorical pipelines
                           to the respective columns.
    """
    logger.info("Starting columns transformation.")

    # Record the start time
    start_time = time.time()
    preprocessor = ColumnTransformer(transformers=[
        ('num', preprocess_numeric_columns(), selector(dtype_include=np.number)),  # Apply numerical pipeline
        ('cat', preprocess_categorical_columns(), selector(dtype_exclude=np.number))  # Apply categorical pipeline
    ])

    # Log the end of the transformation process and the time taken
    end_time = time.time()
    logger.info(f"{GREEN}Transformation completed. Time taken: {end_time - start_time:.4f} seconds{ENDC}")

    return preprocessor


def filter_columns(columns: List[str], exclude_columns: List[str] = None) -> List[str]:
    """
    Filters columns into categorical (object) types, excluding specified columns.

    Args:
        columns (List[str]): List of column names to filter.
        exclude_columns (List[str]): List of column names to exclude.

    Returns:
        The list contains categorical (object) columns.
    """
    start_time = time.time()

    # Check if exclude_columns is not empty or None
    if exclude_columns and exclude_columns[0] is not None:
        # Remove exclude columns from the main list
        filtered_columns = [col for col in columns if col not in exclude_columns]
    else:
        filtered_columns = columns

    end_time = time.time()
    logger.info(f"Filtering columns took {end_time - start_time:.4f} seconds.")

    return filtered_columns
