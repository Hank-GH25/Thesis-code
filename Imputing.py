import pandas as pd

# Load the dataset
file_path = r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\LCI_transformed_ordered.csv"
df = pd.read_csv(file_path)

# Extract country column
countries = df["Country"]

# Convert wide format to datetime index
df = df.set_index("Country").T

# Convert column names to period format
df.index = pd.period_range(start=df.index[0].replace("-Q1", "-01").replace("-Q2", "-04").replace("-Q3", "-07").replace("-Q4", "-10"), periods=len(df.index), freq="Q")

# Resample to monthly and interpolate
df = df.resample("M").interpolate()

# Convert index back to string format (YYYY-MM)
df.index = df.index.strftime("%Y-%m")

# Convert back to original format
df = df.T.reset_index()
df.rename(columns={"index": "Country"}, inplace=True)

# Save the output
output_path = "LCI_monthly_data.csv"
df.to_csv(output_path, index=False)

print(f"Monthly interpolated data saved to {output_path}")
