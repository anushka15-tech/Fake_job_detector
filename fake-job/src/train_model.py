import pandas as pd

# Dataset load karo
df = pd.read_csv('data/fake_job_postings.csv')

# Duplicate records remove karo
df = df.drop_duplicates()

# Null values remove karo (important columns)
df = df.dropna(subset=['title', 'description', 'company_profile', 'requirements', 'benefits'])

# Cleaned data ko save karo
df.to_csv('data/processed.csv', index=False)
print("Processed data saved!")
