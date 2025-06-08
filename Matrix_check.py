import sqlite3
import pandas as pd

# Configuration
DATABASE_NAME = "eu_trade.db"  # Change if your database name is different
OUTPUT_FOLDER = "exported_tables"  # Folder to save the CSV file

def export_table_to_csv(table_name):
    """Exports a specific table from the database to a CSV file."""
    conn = sqlite3.connect(DATABASE_NAME)

    try:
        # Read the table into a Pandas DataFrame
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}";', conn)
        
        # Ensure the output folder exists
        import os
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Define the output file path
        output_file = f"{OUTPUT_FOLDER}/{table_name}.csv"

        # Save DataFrame to CSV
        df.to_csv(output_file, index=False)
        print(f"Table '{table_name}' has been exported to '{output_file}'.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        conn.close()

if __name__ == "__main__":
    table_name = input("Enter the table name to export: ")  # Prompt user for table name
    export_table_to_csv(table_name)
