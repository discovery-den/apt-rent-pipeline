import logging
import time

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI escape codes for colored logging
GREEN = '\033[92m'
ENDC = '\033[0m'


def feature_selection_pipeline(alpha: float = 0.01) -> Pipeline:
    """
    Creates a scikit-learn pipeline for feature selection using SelectFromModel with Lasso for a regression model.

    Args:
        alpha (float, optional): The regularization strength for Lasso. Defaults to 0.01.

    Returns:
        Pipeline: A scikit-learn pipeline for feature selection.
    """
    # Log the start of the pipeline creation process
    logger.info("Starting feature selection pipeline creation.")

    # Record the start time
    start_time = time.time()

    # Create a pipeline for feature selection using Lasso
    fs_pipeline = Pipeline([
        ('feature_selection', SelectFromModel(Lasso(alpha=alpha))),
    ])

    # Log the end of the pipeline creation process and the time taken
    end_time = time.time()
    logger.info(f"{GREEN}Feature selection pipeline created. Time taken: {end_time - start_time:.4f} seconds{ENDC}")

    return fs_pipeline
