import logging
import os
import time

import pandas as pd

from utils import fix_skewness, clean_text, drop_duplicate_info_columns, drop_columns_with_high_missing_values, \
    detect_outliers_zscore, fill_service_charge_with_city_median, handle_total_rent

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
GREEN = '\033[92m'
ENDC = '\033[0m'


def load_sample_file_local(file_name: str) -> pd.DataFrame:
    """
    Loads a sample file from the local "resource/dataset" directory and returns it as a Pandas DataFrame.

    Args:
        file_name (str): The name of the file to be loaded.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the loaded file.

    Raises:
        FileNotFoundError: If the file does not exist in the specified directory.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If the file cannot be parsed.
    """
    # Define the base directory where the files are located
    base_dir = os.path.join(os.path.dirname(__file__), "..", "resource", "dataset")

    # Construct the full path to the file
    file_path = os.path.join(base_dir, file_name)

    # Log the start of the file loading process
    logger.info(f"Loading file: {file_path}")

    # Record the start time
    start_time = time.time()

    try:
        # Load the file into a Pandas DataFrame and drop NA
        df = pd.read_csv(file_path)

        # Log the successful file loading and the time taken
        logger.info(f"Describe Table {df.describe().round(3)}")
        end_time = time.time()
        logger.info(f"{GREEN}File loaded successfully: {file_path}{ENDC}")
        logger.info(f"{GREEN}Time taken to load the file: {end_time - start_time:.4f} seconds{ENDC}")

        # Log the top 5 rows of the DataFrame
        logger.debug("Top 5 rows of the DataFrame:")
        logger.debug(df.head())

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_name} does not exist in the {base_dir} directory.")

    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {file_path}")
        raise pd.errors.EmptyDataError(f"The file {file_name} is empty.")

    except pd.errors.ParserError:
        logger.error(f"Error parsing the file: {file_path}")
        raise pd.errors.ParserError(f"The file {file_name} could not be parsed.")


def clean_df(df: pd.DataFrame, long_text_columns: list = None, drop_long_text_columns: bool = True) -> pd.DataFrame:
    """
    Load a sample CSV file, clean the data, and return a processed DataFrame.

    Parameters:
    file_path (str): The path to the CSV file to be loaded and cleaned.
    long_text_columns (list): List of columns containing long text to be cleaned. Default is None.
    drop_long_text_columns (bool): If True, drop the long text columns after cleaning. Default is True.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    start_time = time.time()
    logging.info(f"cleaning data from {df.info}...")

    # delete columns with duplicate info based on description
    df = drop_duplicate_info_columns(df)

    # drop columns with high missing values
    df = drop_columns_with_high_missing_values(df)

    # drop all the rows where construction year is 2021 and above
    df = df.drop(df[df['yearConstructed'] >= 2021].index)

    # detect outliers using zcore and remove
    df = detect_outliers_zscore(df)

    # fix service charge column, required for totalRent column later
    df = fill_service_charge_with_city_median(df)

    # handle target value
    df = handle_total_rent(df)

    # Fix the skewness
    df = fix_skewness(df)

    # Convert geo_plz into distance using stuttgart as reference
    # Distance conversion didn't work well
    # df = calculate_haversine_distances(df, load_sample_file_local('plz_coordinates.csv'))

    # Calculate the percentage of missing values in each column
    missing_percentage = df.isnull().sum() * 100 / len(df)

    # Log the missing percentage for each column
    logging.debug(f"{GREEN}Missing value percentage for each column: {missing_percentage}{ENDC}")
    # Clean the text columns if provided and not dropping them
    if long_text_columns and not drop_long_text_columns:
        for col in long_text_columns:
            df[col] = df[col].apply(clean_text)

    # Drop long text columns if specified
    if drop_long_text_columns and long_text_columns:
        df = df.drop(columns=long_text_columns)

    end_time = time.time()
    logging.info(f"{GREEN}Data cleaning completed in {end_time - start_time:.2f} seconds.{ENDC}")

    return df

# if __name__ == "__main__":
#     try:
#         df = load_sample_file_local("immo_data.csv")
#         print("Top 5 rows of the DataFrame:")
#         print(df.head())
#     except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
#         print(e)
