import logging
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import functions from respective modules
from feature_selection import feature_selection_pipeline
from load_sample_file import clean_df, load_sample_file_local
from model_generation import lr_model_generation_pipeline, generate_and_save_model, evaluate_model, \
    dtr_model_generation_pipeline
from preprocessing import preprocess_columns, preprocess_columns_with_long_text, filter_columns

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI escape codes for colored logging
GREEN = '\033[92m'
ENDC = '\033[0m'


def model_generation_without_long_text_columns():
    """
    Generates a model without using the "description" and "facilities" text fields.

    This function loads the dataset, preprocesses it, splits it into training and testing sets,
    creates a pipeline that excludes text fields, generates the model, and evaluates its performance.
    """
    # Log the start of the model generation process
    logger.info("Starting model generation without long text columns.")

    # Record the start time
    start_time = time.time()

    # Load the sample file
    long_text_columns = ['description', 'facilities']
    # Define the target column
    target_column = 'totalRent'
    # Load the sample file
    df = load_sample_file_local("immo_data.csv")
    df = clean_df(df=df, long_text_columns=long_text_columns, drop_long_text_columns=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), df[target_column],
                                                        test_size=0.2, random_state=42)

    # Create the final pipeline
    final_pipeline = Pipeline(steps=[
        ('preprocess_columns', preprocess_columns()),
        ('model_generation', dtr_model_generation_pipeline())
    ])

    # Generate and save the model
    model_name = "lr_model_without_long_text_field.pkl"
    generate_and_save_model(final_pipeline, X_train, y_train, model_name)

    # Evaluate the model
    evaluate_model(final_pipeline, X_test, y_test, model_name)

    # Log the end of the model generation process and the time taken
    end_time = time.time()
    logger.info(f"{GREEN}Model generation without long text columns completed. Time taken: {end_time - start_time:.4f} seconds{ENDC}")


def model_generation_with_long_text_columns():
    """
    Generates a model including the "description" and "facilities" text fields.

    This function loads the dataset, preprocesses it, splits it into training and testing sets,
    creates a pipeline that includes text fields, generates the model, and evaluates its performance.
    """
    # Log the start of the model generation process
    logger.info("Starting model generation with long text columns.")

    # Record the start time
    start_time = time.time()

    # Load the sample file
    long_text_columns = ['description', 'facilities']
    # Define the target column
    target_column = 'totalRent'
    # Load the sample file
    df = load_sample_file_local("immo_data.csv").dropna()
    df = clean_df(df=df, long_text_columns=long_text_columns, drop_long_text_columns=False)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), df[target_column],
                                                        test_size=0.2, random_state=42)

    # Create the final pipeline
    category_columns = filter_columns(df.select_dtypes(include=['object']).columns.tolist(), long_text_columns)
    final_pipeline = Pipeline(steps=[
        ('preprocess_with_long_text_columns', preprocess_columns_with_long_text(category_columns, long_text_columns)),
        ('feature_selection', feature_selection_pipeline()),
        ('model_generation', lr_model_generation_pipeline())
    ])

    # Generate and save the model
    model_name = "lr_model_with_long_text_field.pkl"

    generate_and_save_model(final_pipeline, X_train, y_train, model_name)

    # Evaluate the model
    evaluate_model(final_pipeline, X_test, y_test, model_name)

    # Log the end of the model generation process and the time taken
    end_time = time.time()
    logger.info(f"{GREEN}Model generation with long text columns completed. Time taken: {end_time - start_time:.4f} "
                f"seconds{ENDC}")


if __name__ == "__main__":
    model_generation_with_long_text_columns()
    model_generation_without_long_text_columns()
