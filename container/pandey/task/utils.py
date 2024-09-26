import logging
import re
import time
from math import radians

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import PowerTransformer

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
GREEN = '\033[92m'
ENDC = '\033[0m'


def fix_skewness(df: pd.DataFrame, method: str = 'yeo-johnson', threshold: float = 0.5) -> pd.DataFrame:
    """
    Applies PowerTransformer from scikit-learn to handle skewness in numerical columns.

    Args:
        df (pd.DataFrame): The input dataframe.
        method (str): The method of transformation, 'yeo-johnson' (default) or 'box-cox'.
        threshold (float): The skewness threshold above which transformations are applied.

    Returns:
        pd.DataFrame: The dataframe with transformed columns where skewness was high or moderate.
    """
    start_time = time.time()
    logger.info("Starting skewness correction process.")

    df_transformed = df.copy()

    # Select only numerical columns
    # numerical_columns = df.select_dtypes(include=[np.number]).columns
    numerical_columns = ['serviceCharge', 'livingSpace', 'totalRent']
    # Check skewness and apply transformation for skewed columns
    for col in numerical_columns:
        skewness = df[col].skew()
        logger.info(f"Column '{col}' skewness: {skewness:.4f}")

        if abs(skewness) > threshold:
            # Apply PowerTransformer only if skewness is moderate or high
            logger.info(f"Applying {method} transformation to column: {col}")
            pt = PowerTransformer(method=method)
            transformed_col = pt.fit_transform(df[[col]])
            df_transformed[col] = transformed_col

    end_time = time.time()
    logger.info(f"{GREEN}Skewness correction completed. Time taken: {end_time - start_time:.4f} seconds{ENDC}")

    return df_transformed


def calculate_haversine_distances(
        df: pd.DataFrame, plz_coordinates: pd.DataFrame, lat_column='lat', lon_column='lon'
) -> pd.DataFrame:
    """
    Calculates the Haversine distance in kilometers between each coordinate in the DataFrame and the reference
    coordinates of Stuttgart.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the latitude and longitude columns.
    lat_column (str): The name of the column containing the latitude values. Default is 'lat'.
    lon_column (str): The name of the column containing the longitude values. Default is 'lon'.

    Returns:
    pd.DataFrame: The DataFrame with an additional 'distance' column containing the Haversine distances in kilometers.
    """
    # Start timing
    start_time = time.time()
    logging.info("Starting Haversine distance calculation...")

    # Convert reference coordinates of stuttgart to radians
    ref_lat_rad = radians(48.7758)
    ref_lon_rad = radians(9.1829)

    # merge it with main data frame
    df = pd.merge_asof(df.sort_values('geo_plz'), plz_coordinates.sort_values('geo_plz'), left_on='geo_plz',
                       right_on='geo_plz', direction='nearest'
                       )

    # Calculate Haversine distance for each coordinate from Stuttgart
    def calculate_distance(row):
        lat_rad = radians(row[lat_column])
        lon_rad = radians(row[lon_column])
        coord = [lat_rad, lon_rad]
        ref_coord = [ref_lat_rad, ref_lon_rad]
        distance = haversine_distances([coord, ref_coord]) * 6371  # Multiply by Earth radius in km
        return distance[0][1]

    # Apply the distance calculation function to each row
    df['distance'] = df.apply(calculate_distance, axis=1)

    # Drop all the geo related columns and keep only distance columns
    df.drop(['geo_plz', 'regio2', 'lon', 'lat'], axis=1)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Haversine distance calculation completed in {elapsed_time:.2f} seconds.")

    return df


def detect_outliers_zscore(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    """
    Detects outliers in numerical columns using the Z-Score method and prints an outlier matrix.

    Args:
        df (pd.DataFrame): Input dataframe.
        threshold (float): Z-score value beyond which data is considered an outlier. Default is 3.

    Returns:
        pd.DataFrame: Boolean DataFrame where True indicates an outlier.
        pd.Series: Number of outliers in each column.
    """
    start_time = time.time()

    # Select numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # Calculate Z-scores for numerical columns
    z_scores = np.abs(stats.zscore(df[numerical_columns]))

    # Create a boolean DataFrame indicating outliers
    outliers_matrix = pd.DataFrame(z_scores > threshold, columns=numerical_columns)

    # Count the number of outliers per column
    outlier_counts = outliers_matrix.sum()

    # Log the number of outliers per column
    logging.info("Number of outliers in each column (Z-Score Method):")
    logging.info(outlier_counts)

    # Drop rows with outliers
    df_cleaned = df[~outliers_matrix.any(axis=1)]

    # Log the shape of the cleaned DataFrame
    logging.info(f"Shape of the cleaned DataFrame: {df_cleaned.shape}")

    end_time = time.time()
    logging.info(f"Time taken to detect outliers: {end_time - start_time:.2f} seconds")

    return df_cleaned


def drop_columns_with_high_missing_values(df: pd.DataFrame, threshold: int = 35) -> pd.DataFrame:
    """
    Calculate the percentage of missing values in each column and drop columns where the percentage is above a specified threshold.

    Args:
        df (pd.DataFrame): Input dataframe.
        threshold (float): Threshold percentage for missing values. Columns with missing values above this threshold will be dropped. Default is 35.

    Returns:
        pd.DataFrame: DataFrame with columns having missing values above the threshold dropped.
    """
    start_time = time.time()

    # Calculate the percentage of missing values in each column
    missing_percentage = df.isnull().sum() * 100 / len(df)

    # Log the missing percentage for each column
    logging.info("Missing value percentage for each column:")
    logging.info(missing_percentage)

    # Identify columns to drop
    columns_to_drop = missing_percentage[missing_percentage > threshold].index

    # Drop the identified columns
    df_cleaned = df.drop(columns=columns_to_drop)

    # Log the columns that were dropped
    logging.info(f"Columns dropped due to high missing value percentage (> {threshold}%):")
    logging.info(columns_to_drop)

    end_time = time.time()
    logging.info(f"Time taken to drop columns: {end_time - start_time:.4f} seconds")

    return df_cleaned


def drop_duplicate_info_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop specified columns from the DataFrame that contain duplicate information based on column description.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """
    start_time = time.time()

    # List of columns to drop
    delete_cols = [
        'scoutId', 'date', 'regio1', 'regio3', 'geo_bln', 'geo_krs', 'street', 'streetPlain',
        'houseNumber', 'firingTypes', 'thermalChar', 'telekomUploadSpeed', 'telekomHybridUploadSpeed',
        'telekomTvOffer', 'energyEfficiencyClass', 'electricityBasePrice', 'electricityKwhPrice',
        'livingSpaceRange', 'noRoomsRange', 'baseRentRange', 'yearConstructedRange'
    ]

    # Drop the specified columns
    df_cleaned = df.drop(delete_cols, axis=1)

    # Log the columns that were dropped
    logging.info(f"Columns dropped: {delete_cols}")

    end_time = time.time()
    logging.info(f"Time taken to drop columns: {end_time - start_time:.4f} seconds")

    return df_cleaned


def fill_service_charge_with_city_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the 'serviceCharge' column with the median service charge for each city (regio2).

    Args:
        df (pd.DataFrame): Input DataFrame containing 'regio2' and 'serviceCharge' columns.

    Returns:
        pd.DataFrame: DataFrame with missing 'serviceCharge' values filled using city median.
    """
    start_time = time.time()

    # Create a copy of the relevant columns
    service_full = df[['regio2', 'serviceCharge']].copy()

    # Calculate the median service charge for each city
    city_median_service = service_full.groupby('regio2')['serviceCharge'].transform('median')

    # Fill missing values in 'serviceCharge' with the city median
    service_full['serviceCharge'] = service_full['serviceCharge'].fillna(city_median_service)

    # Update the main DataFrame with the filled 'serviceCharge' values
    df['serviceCharge'] = service_full['serviceCharge']

    # Log the number of missing values filled
    missing_values_filled = df['serviceCharge'].isnull().sum()
    logging.info(f"Number of missing 'serviceCharge' values filled: {missing_values_filled}")

    end_time = time.time()
    logging.info(f"Time taken to fill 'serviceCharge' with city median: {end_time - start_time:.4f} seconds")

    return df


def handle_total_rent(df: pd.DataFrame, tolerance: float = 5) -> pd.DataFrame:
    """
    Impute missing 'totalRent' values by summing 'baseRent' and 'serviceCharge', and drop rows with negative residue.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'totalRent', 'baseRent', and 'serviceCharge' columns.
        tolerance (float): Tolerance value for residue calculation. Default is 5.

    Returns:
        pd.DataFrame: DataFrame with missing 'totalRent' values imputed and rows with negative residue dropped.
    """
    start_time = time.time()

    # Create a copy of the relevant columns
    total_rent_df = df[['totalRent', 'baseRent', 'serviceCharge']].copy()

    # Identify indices where 'totalRent' is NaN
    totalrent_nan_idx = total_rent_df[total_rent_df['totalRent'].isna()].index

    # Calculate the residue of total rent - (base rent + service charge)
    total_rent_df['residue'] = total_rent_df['totalRent'] - (total_rent_df['baseRent'] + total_rent_df['serviceCharge'])

    # Compute the percentage of exact, positive, and negative residue
    poss = total_rent_df['residue'][total_rent_df['residue'] > tolerance].count() / total_rent_df['residue'].shape[0]
    neg = total_rent_df['residue'][total_rent_df['residue'] < -tolerance].count() / total_rent_df['residue'].shape[0]
    exact = 1 - poss - neg

    logging.info(f'Percentage of zero residue: {100 * exact:.2f} %')
    logging.info(f'Percentage of positive residue: {100 * poss:.2f} %')
    logging.info(f'Percentage of negative residue: {100 * neg:.2f} %')

    # Sum values of base rent and service charge at indices where 'totalRent' is NaN
    rent_sum_val = total_rent_df.loc[totalrent_nan_idx, 'baseRent'] + total_rent_df.loc[
        totalrent_nan_idx, 'serviceCharge']

    # Recompute the positive residue rate after dropping negative residue values
    poss_new = total_rent_df['residue'][total_rent_df['residue'] > tolerance].count() / total_rent_df['residue'].shape[
        0]

    # Add the median of positive residue as additional costs for the positive fraction
    poss_fraction = rent_sum_val.sample(frac=poss_new, random_state=2024)
    rent_sum_val.loc[poss_fraction.index] += total_rent_df['residue'][total_rent_df['residue'] > tolerance].median()

    # Fill NaN values in 'totalRent' with the sum of 'baseRent' and 'serviceCharge'
    total_rent_df.loc[totalrent_nan_idx, 'totalRent'] = rent_sum_val

    # Identify indices with negative residue
    neg_index = total_rent_df['residue'][total_rent_df['residue'] < -tolerance].index
    logging.info(f'Number of rows will be dropped: {len(neg_index)}')

    # Update the main DataFrame with the imputed 'totalRent' values
    df['totalRent'] = total_rent_df['totalRent']

    # Drop rows with negative residue
    df = df.drop(neg_index)

    end_time = time.time()
    logging.info(f"Time taken to impute and drop 'totalRent': {end_time - start_time:.4f} seconds")

    return df


def clean_text(text) -> str:
    """
    Clean the text data by removing unwanted characters and formatting.

    Parameters:
    text (str): The text to be cleaned.

    Returns:
    str: The cleaned text. If the input is NaN, returns an empty string.
    """

    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)

    # Remove special characters and numbers (optional)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
