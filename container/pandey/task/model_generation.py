import json
import logging
import os
import time

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# from xgboost import XGBRegressor

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI escape codes for colored logging
GREEN = '\033[92m'
ENDC = '\033[0m'


def lr_model_generation_pipeline() -> Pipeline:
    """
    Creates a scikit-learn pipeline for model generation using LinearRegression.

    Returns:
        Pipeline: A scikit-learn pipeline for model generation.
    """
    # Log the start of the pipeline creation process
    logger.info("Starting basic model generation pipeline creation.")

    # Record the start time
    start_time = time.time()

    # Optimized parameters for LinearRegression
    optimized_params = {
        'fit_intercept': True,
        'positive': False,
        'copy_X': True,
        'n_jobs': None
    }
    # Create a pipeline for model generation using LinearRegression with optimized parameters
    simple_model_pipeline = Pipeline([
        ('lr_model_gen', LinearRegression(**optimized_params))
    ])

    # Log the end of the pipeline creation process and the time taken
    end_time = time.time()
    logger.info(
        f"{GREEN}Basic model generation pipeline created using LinearRegression. Time taken: {end_time - start_time:.4f} seconds{ENDC}")

    return simple_model_pipeline


def dtr_model_generation_pipeline() -> Pipeline:
    """
    Creates a scikit-learn pipeline for model generation using DecisionTreeRegressor.

    Returns:
        Pipeline: A scikit-learn pipeline for model generation.
    """
    # Log the start of the pipeline creation process
    logger.info("Starting basic model generation pipeline creation.")

    # Record the start time
    start_time = time.time()

    # Optimized parameters for DecisionTreeRegressor
    optimized_params = {
        'criterion': 'squared_error',  # Example optimized criterion
        'max_depth': 5,  # Example optimized maximum depth
        'min_samples_split': 10,  # Example optimized minimum samples to split
        'min_samples_leaf': 5,  # Example optimized minimum samples per leaf
        'random_state': 42  # Random seed for reproducibility
    }

    # Create a pipeline for model generation using DecisionTreeRegressor with optimized parameters
    simple_model_pipeline = Pipeline([
        ('dtr_model_gen', DecisionTreeRegressor(**optimized_params))
    ])

    # Log the end of the pipeline creation process and the time taken
    end_time = time.time()
    logger.info(
        f"{GREEN}Basic model generation pipeline created using DecisionTreeRegressor. Time taken: {end_time - start_time:.4f} seconds{ENDC}")

    return simple_model_pipeline


def generate_and_save_model(model_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_name: str):
    """
    Trains the model and saves it to the specified path.

    Args:
        model_pipeline (Pipeline): The scikit-learn pipeline for model generation.
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        model_name (str): The model name.
    """
    # Log the start of the model training process
    logger.info("Starting model training.")

    # Record the start time
    start_time = time.time()

    # Fit the model
    model_pipeline.fit(X_train, y_train)

    # Save the model
    base_dir = os.path.join(os.path.dirname(__file__), "..", "resource", "models")

    # Construct the full path to the file
    model_path = os.path.join(base_dir, model_name)

    joblib.dump(model_pipeline, model_path)

    # Log the end of the model training process and the time taken
    end_time = time.time()
    logger.info(f"{GREEN}Model trained and saved. Time taken: {end_time - start_time:.4f} seconds{ENDC}")


def evaluate_model(model_pipeline, X_test, y_test, model_name):
    """
    Evaluates the model and saves all possible metrics including feature importance to a JSON file.

    Args:
        model_pipeline (Pipeline): The scikit-learn pipeline for model generation.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target.
        model_name (str): The name of the model, used to name the output JSON file.
    """
    # Log the start of the model evaluation process
    logger.info("Starting model evaluation.")

    # Record the start time
    start_time = time.time()

    # Predict on the test set
    y_pred = model_pipeline.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Prepare metrics dictionary
    metrics = {
        "model_name": model_name,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "r2_score": r2,
        "evaluation_time": time.time() - start_time
    }

    # Log the metrics
    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"Root Mean Squared Error: {rmse}")
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"R^2 Score: {r2}")

    # Define the output directory and file path
    # Define the base directory where the files are located
    base_dir = os.path.join(os.path.dirname(__file__), "..", "resource", "results")

    # Construct the full path to the file
    os.makedirs(base_dir, exist_ok=True)  # Ensure the directory exists
    version = 2
    output_file = os.path.join(base_dir, f"{model_name}_{version}.json")

    # Save metrics to JSON file
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Log the end of the model evaluation process and the time taken
    logger.info(f"Model evaluation completed. Time taken: {time.time() - start_time:.4f} seconds")
    logger.info(f"Metrics saved to {output_file}")
