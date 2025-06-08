import sqlite3
import pandas as pd
import numpy as np
import os

# SQLite database file
DB_PATH = "eu_trade.db"

# List of EU countries (to match SQL table names)
eu_countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France", 
    "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", 
    "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Dictionary to store spatial weight matrices
w_dict = {}

# Get a list of time periods from one table (assuming all tables have same structure)
cursor.execute(f"SELECT * FROM {eu_countries[0]}_tot LIMIT 1")
columns = [desc[0] for desc in cursor.description][1:]  # Skip first column (country names)

def process_trade_flows():
    global w_dict

    # Read all trade data once and store in a dictionary for faster access
    trade_data = {}
    for country in eu_countries:
        query = f"SELECT * FROM {country}_tot"
        df = pd.read_sql_query(query, conn)
        
        # Rename first column to 'Country'
        df.rename(columns={df.columns[0]: "Country"}, inplace=True)

        # Replace errors with NaN and convert trade values to numeric
        df.replace("#VALUE!", np.nan, inplace=True)

        # Store this dataframe in the dictionary
        trade_data[country] = df

    # Now iterate through time periods to construct weight matrices
    for time_period in columns:
        w_matrix = np.zeros((27, 27))  # Empty 27x27 matrix

        for col_idx, col_country in enumerate(eu_countries):  
            df = trade_data[col_country]  # Get the correct country's trade data
            if time_period not in df.columns:
                continue  # Skip if time period column is missing

            # Convert trade values to numeric
            df[time_period] = pd.to_numeric(df[time_period], errors="coerce")

            # Compute total trade for this country
            col_total = df[time_period].sum()
            if col_total > 0:
                df["Weight"] = df[time_period] / col_total
            else:
                df["Weight"] = 0  # Avoid division by zero

            # Assign values to the correct column of the matrix
            for row_idx, row_country in enumerate(eu_countries):
                trade_weight = df.loc[df["Country"] == row_country, "Weight"].values
                if trade_weight.size > 0:
                    w_matrix[row_idx, col_idx] = trade_weight[0]

        # Store matrix in dictionary
        w_dict[time_period] = w_matrix

        # Save matrix to SQL
        matrix_df = pd.DataFrame(w_matrix, index=eu_countries, columns=eu_countries)
        matrix_df.to_sql(f"W_{time_period}", conn, if_exists="replace", index=True)

        # Debugging: Print first few rows of the matrix
        print(f"Spatial Weight Matrix for {time_period} (first 5 rows & cols):")
        print(matrix_df.iloc[:5, :5])  


# Run the function
process_trade_flows()


# Close database connection
conn.close()

# The w_dict is now ready for use in PySAL models
