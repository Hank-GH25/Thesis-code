import pandas as pd
import sqlite3

# ==========================
# 🔹 Step 1: Load CSV Files
# ==========================
# List of CSV files (one per variable)
csv_files = {
    "y": r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\HICP_transformed_ordered.csv",   # Dependent variable
    "x1": r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Unemployment_transformed_ordered.csv",   # Independent variable 1
    "x2": r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\LCI_monthly_data.csv"    # Independent variable 2
}

# Initialize an empty list to store transformed data
panel_data = []

for var_name, file_name in csv_files.items():
    # Load CSV
    df = pd.read_csv(file_name)

    # Convert from wide to long format
    df_long = df.melt(id_vars=["Country"], var_name="time", value_name=var_name)

    # Convert time to datetime format
    df_long["time"] = pd.to_datetime(df_long["time"], format="%Y-%m")

    # Append to panel data list
    panel_data.append(df_long)

# ==========================
# 🔹 Step 2: Merge & Process Data
# ==========================
# Merge all variables into a single panel dataframe
df_panel = panel_data[0]  # Start with first variable (y)
for i in range(1, len(panel_data)):  
    df_panel = df_panel.merge(panel_data[i], on=["Country", "time"])

# Assign a numeric region ID (if not already present)
df_panel["region_id"] = df_panel["Country"].astype("category").cat.codes  # Convert Country to numeric

# Sort data
df_panel = df_panel.sort_values(["region_id", "time"])

# Convert Period to string before saving to SQL
df_panel["time"] = df_panel["time"].astype(str)

# Display transformed panel data
print("📊 Transformed Panel Data:")
print(df_panel.head())

# ==========================
# 🔹 Step 3: Save to SQLite
# ==========================
# Path to your existing SQL database
db_path = "eu_trade.db"  # Update this with the actual database name

# Connect to the database
conn = sqlite3.connect(db_path)

# Store panel data in the existing database
df_panel.to_sql("panel_data", conn, if_exists="replace", index=False)

# ==========================
# 🔹 Step 4: Check Column Names in SQL
# ==========================
print("\n✅ Checking Column Names in SQL Database...")
query = "PRAGMA table_info(panel_data);"
columns_in_db = pd.read_sql(query, conn)

print("📋 Columns in 'panel_data' table:")
print(columns_in_db)

# Close connection
conn.close()

print("\n🎉 Panel data successfully uploaded to SQL database!")
