import pandas as pd

# Load the CSV file
file_path = r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\LCI_data_2014_2024.csv"  # Update with the correct file path
df = pd.read_csv(file_path)

# Pivot the dataframe
df_pivot = df.pivot(index="Country", columns="TIME_PERIOD", values="OBS_VALUE")

# Drop the United Kingdom row
df_pivot = df_pivot.drop(index="United Kingdom", errors="ignore")

# Define the desired country order
country_order = [
    "Austria", "Belgium", "Bulgaria", "Cyprus", "Czechia", "Germany", "Denmark",
    "Estonia", "Spain", "Finland", "France", "Greece", "Croatia", "Hungary",
    "Ireland", "Italy", "Lithuania", "Luxembourg", "Latvia", "Malta",
    "Netherlands", "Poland", "Portugal", "Romania", "Sweden", "Slovenia", "Slovakia"
]

# Reorder the DataFrame based on the given country order
df_pivot = df_pivot.reindex(country_order)

# Save the transformed dataframe to a new CSV file
output_file_path = "LCI_transformed_ordered.csv"  # Update with the desired output path
df_pivot.to_csv(output_file_path)

print(f"Transformed data saved to {output_file_path}")
