import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/loan_data_new.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Basic information
print("=" * 50)
print("LOAN PREDICTION - EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nLoan Status distribution:")
print(df["Loan Status"].value_counts())

# 1. Loan status distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Loan Status")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status (0 or 1)")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.savefig("outputs/loan_status_distribution.png")
plt.close()

# 2. Income distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Person Income", bins=40, kde=True)
plt.title("Person Income Distribution")
plt.tight_layout()
plt.savefig("outputs/income_distribution.png")
plt.close()

# 3. Credit score distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Credit Score", bins=30, kde=True)
plt.title("Credit Score Distribution")
plt.tight_layout()
plt.savefig("outputs/credit_score_distribution.png")
plt.close()

# 4. Loan status by loan intent
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Loan Intent", hue="Loan Status")
plt.title("Loan Status by Loan Intent")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("outputs/loan_status_by_intent.png")
plt.close()

print("\nEDA completed successfully!")
print("Charts saved in the outputs folder.")
 
