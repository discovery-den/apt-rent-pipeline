# Apartment Rental Offers Prediction Pipeline

This project processes apartment rental offers in Germany using machine learning pipelines. It involves loading datasets, preprocessing data, and using transformer-based models to predict rental prices (`totalRent`). 

# Model Evaluation Results

This section provides a summary of the model performance results. The details of model creation, data preprocessing, and cleaning can be found in the sections below.

## Model 1: `Predicting Rent with Structural Data:`

Develop a machine learning model to predict the total rent using only the structural data. Exclude the “description” and “facilities” text fields for this model.

```json
{
   "model_name": "lr_model_without_long_text_field.pkl",
   "mean_squared_error": 0.03446195446055379,
   "root_mean_squared_error": 0.18563931280995896,
   "mean_absolute_error": 0.11813638951194545,
   "r2_score": 0.9650713694667908,
   "evaluation_time": 0.0770101547241211
}
```

## Model 2: `Predicting Rent with Structural and Text Data`

Create a second machine learning model that predicts the total rent using both structural and text data (“description” and “facilities”). We encourage using modern generative AI techniques for processing text data.

```json
{
   "model_name": "lr_model_with_long_text_field.pkl",
   "mean_squared_error": 0.07451682325927053,
   "root_mean_squared_error": 0.27297769736604954,
   "mean_absolute_error": 0.21600995861455072,
   "r2_score": 0.9326289759958837,
   "evaluation_time": 0.4592320919036865
}
```

## Directory Structure

The directory structure is designed for easy navigation and reproducibility.

<pre>
├── Dockerfile               # Docker configuration file for building the project container
├── LICENSE                  # License file
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies for the project
├── .github
│   └── workflows
│        └──docker-image.yml # Build a Docker image to deploy in registry
└── container
   └── pandey
      ├── resource
      │   ├── dataset      # Contains datasets used in the project
      │   ├── models       # Folder to store trained machine learning models
      │   └── results      # Folder to store output results
      └── task
          ├── BERTTransformer.py        # Code for BERT-based text transformation
          ├── ELECTRATransformer.py     # Code for ELECTRA-based text transformation
          ├── FlairTransformer.py       # Code for Flair-based text transformation
          ├── feature_selection.py      # Feature selection logic
          ├── load_sample_file.py       # Load and preprocess the sample CSV dataset
          ├── model_generation.py       # Code for model training and saving models
          ├── pipeline.py               # Main pipeline to preprocess data and run the model
          └── preprocessing.py          # Code for data preprocessing
</pre>


## Setup Instructions

### Step 1: Prepare Datasets

1. **Create a `dataset` folder**:
    - Inside `container/pandey/resource`, create a folder named `dataset`.

2. **Download Dataset**:
    - Download the dataset file (`immo_data.csv`) from [Kaggle's Apartment Rental Offers in Germany Dataset](https://www.kaggle.com/datasets/corrieaar/apartment-rental-offers-in-germany/data).
    - Save the `immo_data.csv` file inside the `dataset` folder.

3. **Create a `models` folder**:
    - Inside `container/pandey/resource`, create a folder named `models` for storing the trained machine learning models.

### Step 2: Install Dependencies

Ensure Docker is installed on your system. The project uses `requirements.txt` to manage dependencies, which will be installed inside the Docker container.

### Step 3: Building the Docker Image

You can build the Docker image for the project using one of the following methods:

#### Option 1: Use GitHub Actions (CI/CD Preferred)

You can configure a GitHub Actions workflow to automatically build the Docker image and push it to GitHub Packages.

#### Option 2: Build Locally

To build the Docker image locally, navigate to the root directory where the `Dockerfile` is located and run the following command:

```bash
docker build -t rent_prediction:0.1 .
docker run rent_prediction:0.1
```
## Data Preprocessing and Feature Engineering

This section outlines the key steps involved in preparing and transforming the dataset for model training.

### Skewness Correction
Applies power transformations to reduce skewness in numerical columns when the skewness exceeds a threshold of 3.

### Haversine Distance Calculation
Convert all location data to the distance from them to Stuttgart using the Haversine distance between two geographic points (latitude, longitude) to capture spatial relationships.

### Outlier Detection (Z-Score)
Identifies and removes outliers using the Z-score method, where values exceeding a threshold of 0.5.

### Dropping Columns with High Missing Values
Drops columns with a high percentage of missing values, above a 35% threshold.

### Removing Duplicate Information Columns
Removes columns that are redundant or contain duplicate information to avoid data redundancy. Mainly based on column description

### Filling Service Charge with City Median
Fills missing values in the 'serviceCharge' column with the median value for each city to maintain consistency.

### Handling Total Rent Imputation
Imputes missing 'totalRent' values by summing 'baseRent' and 'serviceCharge' and removes rows with invalid data.

### Text Data Cleaning
Cleans text data by removing URLs, special characters, and excess whitespace to improve text feature usability and applied BERT Transformer using model <b>bert-base-german-cased</b>

### Model Generation
For Model generation **LinearRegression** and **DecisionTreeRegressor** were used. In case of Task 1. without text data DecisionTreeRegressor gave best results where in case of Task 2 with text data LinearRegression gave the best results