# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/pal14/VisitWithUs/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
# Dropping customer interaction data as the prediction has to be made before calling the customer
df.drop(columns=['CustomerID', '0', 'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups', 'DurationOfPitch'], inplace=True)

# Convert object columns as categorical columns
df[df.select_dtypes(include='object').columns] = (
    df.select_dtypes(include='object').astype('category')
)

# Replace 'Fe Male' to 'Female' in Gender column
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

# Define target variable
target_col = 'ProdTaken'

# Define numeric and categorical features
numeric_features = [
    'Age', 'CityTier',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation'
]

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="pal14/visitWithUs",
        repo_type="dataset",
    )
