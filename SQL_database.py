import pandas as pd
import sqlite3

# Configuration
DATABASE_NAME = "eu_trade.db"  # SQLite database file

# Manually specify the file paths
CSV_FILES = [
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Austria_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Belgium_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Bulgaria_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Croatia_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Cyprus_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Czechia_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Denmark_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Estonia_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Finland_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\France_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Germany_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Greece_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Hungary_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Ireland_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Italy_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Latvia_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Lithuania_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Luxembourg_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Malta_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Netherlands_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Poland_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Portugal_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Romania_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Slovakia_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Slovenia_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Spain_tot.csv",
    r"C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\Country_trade_files\Sweden_tot.csv",
    # Add more file paths here...
]

def upload_csv_to_sql(file_path):
    """Reads a CSV file and uploads it as a table to the SQLite database."""
    table_name = file_path.split("\\")[-1].replace(".csv", "")  # Extract table name from file name
    
    # Read the CSV file with proper settings
    df = pd.read_csv(file_path, delimiter=",", encoding="utf-8", header=0)  # Adjust delimiter if needed
    df.dropna(axis=1, how="all", inplace=True)  # Remove completely empty columns
    
    # Convert numeric columns
    for col in df.columns[1:]:  
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Connect to the database
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Create table dynamically
    columns = ", ".join([f'"{col}" TEXT' if df[col].dtype == 'object' else f'"{col}" REAL' for col in df.columns])
    cursor.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns});')

    # Insert data into the table
    placeholders = ", ".join(["?" for _ in df.columns])
    query = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
    cursor.executemany(query, df.where(pd.notna(df), None).values.tolist())

    conn.commit()
    conn.close()

    print(f"Uploaded {file_path} to table {table_name} successfully.")

if __name__ == "__main__":
    for file in CSV_FILES:
        upload_csv_to_sql(file)

    print("All specified files uploaded successfully.")
