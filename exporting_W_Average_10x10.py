import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect("eu_trade.db")

# Read the matrix table
df = pd.read_sql_query("SELECT * FROM W_Average", conn)

# Optional: set index to first column if it's country names
# df.set_index(df.columns[0], inplace=True)

# Round to 5 decimal places
df = df.round(5)

# Get the top 10×10 submatrix
sub_df = df.iloc[:10, :10]  # Rows and columns 0–9

# Create figure
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust size for clarity

# Hide axes
ax.axis("off")
ax.axis("tight")

# Create table
table = ax.table(cellText=sub_df.values,
                 colLabels=sub_df.columns,
                 rowLabels=sub_df.index,
                 loc="center",
                 cellLoc="center")

# Adjust font size and cell spacing
for cell in table.get_celld().values():
    cell.set_fontsize(11)

table.scale(1.2, 1.5)

# Save as PDF
plt.tight_layout()
plt.savefig("trade_matrix_10x10.pdf", bbox_inches="tight")

# Close connection
conn.close()
