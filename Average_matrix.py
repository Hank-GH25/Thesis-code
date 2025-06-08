import sqlite3
import pandas as pd
import numpy as np

# --- CONFIG ---
DB_PATH = r'C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\eu_trade.db'
OUTPUT_TABLE = 'W_Average'

# Connect
conn = sqlite3.connect(DB_PATH)

# List all table names
years = range(2014, 2025)
months = range(1, 13)
table_names = [f"W_{year}_{month:02d}" for year in years for month in months]

# Prepare for loading
matrices = []
row_labels = None
col_labels = None

# Load matrices
for table in table_names:
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn, index_col='index')
        
        if row_labels is None:
            row_labels = df.index.tolist()
            col_labels = df.columns.tolist()
        else:
            if row_labels != df.index.tolist() or col_labels != df.columns.tolist():
                raise ValueError(f"Mismatch in labels at table {table}.")

        matrices.append(df.values)
    except Exception as e:
        print(f"Warning: skipping {table}: {e}")

# Check we have matrices
if not matrices:
    raise ValueError("No matrices were loaded!")

# ✅ Step 1: Average ignoring NaNs
average_matrix = np.nanmean(matrices, axis=0)

# ✅ Step 2: Normalize columns, not rows
col_sums = average_matrix.sum(axis=0, keepdims=True)
nonzero_mask = (col_sums != 0)

# Normalize only nonzero columns
average_matrix[:, nonzero_mask[0]] /= col_sums[0, nonzero_mask[0]]

# For zero-sum columns (if any), fill uniformly
zero_cols = (~nonzero_mask[0])
if zero_cols.any():
    print(f"⚠️ {zero_cols.sum()} columns had zero sum after averaging. Filling uniformly.")
    n_rows = average_matrix.shape[0]
    average_matrix[:, zero_cols] = np.full((n_rows, zero_cols.sum()), 1.0 / n_rows)

# ✅ Step 3: Extra tiny fix: force perfect normalization
average_matrix /= average_matrix.sum(axis=0, keepdims=True)

# ✅ Step 4: Final check with relaxed tolerance
col_sums_final = average_matrix.sum(axis=0)
max_deviation = np.abs(col_sums_final - 1).max()
print(f"✅ Max deviation from 1 after column normalization: {max_deviation:.2e}")

if max_deviation > 1e-6:
    raise ValueError("❌ Some columns still deviate from 1 too much!")

# Save to database
average_df = pd.DataFrame(average_matrix, index=row_labels, columns=col_labels)

# Drop if exists
conn.execute(f"DROP TABLE IF EXISTS {OUTPUT_TABLE}")
average_df.to_sql(OUTPUT_TABLE, conn, index=True, index_label='index')

conn.commit()
conn.close()

print(f"✅ Column-normalized average matrix saved to table {OUTPUT_TABLE}.")
