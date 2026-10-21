
# ============================================================
# PRACTICAL: TELECOM TOWER FAILURE
# DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================


# Cell 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")


# Cell 2: Upload the Dataset

from google.colab import files

uploaded = files.upload()


# Cell 3: Load the Dataset

df = pd.read_csv("telecom_tower_failure.csv")

print("\nFirst Five Rows:")
display(df.head())


# Cell 4: Check Dataset Size

print("\nDataset Shape:", df.shape)
print("Number of Records:", df.shape[0])
print("Number of Columns:", df.shape[1])


# Cell 5: Check Column Names

print("\nColumn Names:")
print(df.columns.tolist())


# Cell 6: Understand Data Types

print("\nDataset Information:")
df.info()


# Cell 7: Statistical Summary

print("\nStatistical Summary:")
display(df.describe())


# Cell 8: Check Missing Values

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


# Cell 9: Handle Missing Numerical Values

# Do not impute missing target values.
# Remove rows with missing target values separately.

target = "Failure_Within_48Hrs"

df = df.dropna(subset=[target]).copy()

numeric_columns = df.select_dtypes(
    include=np.number
).columns.drop(target, errors="ignore")

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# Cell 10: Create High Temperature Feature

# Temperature > 60°C = 1, otherwise = 0

df["High_Temperature"] = np.where(
    df["Temperature_C"] > 60, 1, 0
)

print("\nHigh Temperature Feature:")
display(df[["Temperature_C", "High_Temperature"]].head())


# Cell 11: Create Low Battery Feature

# Battery Voltage < 45V = 1, otherwise = 0

df["Low_Battery"] = np.where(
    df["Battery_Voltage"] < 45, 1, 0
)

print("\nLow Battery Feature:")
display(df[["Battery_Voltage", "Low_Battery"]].head())


# Cell 12: Create Power-Traffic Ratio

# Add 0.01 to avoid division by zero

df["Power_Traffic_Ratio"] = (
    df["Power_Consumption_W"] /
    (df["Traffic_Load"] + 0.01)
)

print("\nPower-Traffic Ratio:")
display(
    df[
        [
            "Power_Consumption_W",
            "Traffic_Load",
            "Power_Traffic_Ratio"
        ]
    ].head()
)


# Cell 13: Create Tower Age Category

# Age <= 3 years: New
# Age > 3 and <= 7 years: Medium
# Age > 7 years: Old

df["Tower_Age_Category"] = pd.cut(
    df["Tower_Age_Years"],
    bins=[-np.inf, 3, 7, np.inf],
    labels=["New", "Medium", "Old"]
)

print("\nTower Age Categories:")
display(df[["Tower_Age_Years", "Tower_Age_Category"]].head())


# Cell 14: Apply One-Hot Encoding

df = pd.get_dummies(
    df,
    columns=["Tower_Age_Category"],
    dtype=int
)

print("\nDataset After One-Hot Encoding:")
display(df.head())


# Cell 15: Remove Tower ID

df = df.drop(columns=["Tower_ID"])

print("\nDataset After Removing Tower_ID:")
display(df.head())


# Cell 16: Separate Features (X) and Target (y)

X = df.drop(columns=["Failure_Within_48Hrs"])
y = df["Failure_Within_48Hrs"]

print("\nInput Features (X):")
display(X.head())

print("\nTarget Variable (y):")
display(y.head())


# Cell 17: Check Dimensions of X and y

print("\nNumber of Features:", X.shape[1])
print("Number of Records in X:", X.shape[0])
print("Number of Records in y:", y.shape[0])


# Cell 18: Check and Visualize Target Distribution

print("\nTarget Distribution:")
print(y.value_counts().sort_index())

y.value_counts().sort_index().plot(kind="bar")

plt.xlabel("Failure Within 48 Hours (0 = No, 1 = Yes)")
plt.ylabel("Number of Towers")
plt.title("Tower Failure Distribution")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Cell 19: Save the Feature-Engineered Dataset

output_file = "telecom_tower_feature_engineered.csv"

df.to_csv(output_file, index=False)

print("\nProcessed Dataset Saved Successfully!")
print("File Name:", output_file)

display(df.head())


# Cell 20: Download the Processed Dataset

import os

if os.path.exists(output_file):
    print("\nDownloading Processed Dataset...")
    files.download(output_file)
else:
    print("File not found. Please check Cell 19.")

print("\nPRACTICAL COMPLETED SUCCESSFULLY!")
