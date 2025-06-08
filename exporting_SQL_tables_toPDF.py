import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect("eu_trade.db")

# Read the matrix table
df = pd.read_sql_query("SELECT * FROM W_Average", conn)

# Optional: set index to first column if needed
# df.set_index(df.columns[0], inplace=True)

# Round values to 5 decimal places
df = df.round(5)

# Create figure and axis
fig, ax = plt.subplots(figsize=(22, 22))  # Increase figure size as needed

# Hide axes
ax.axis("off")
ax.axis("tight")

# Create table with custom font size
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 rowLabels=df.index,
                 loc="center",
                 cellLoc="center")

# Increase font size
for key, cell in table.get_celld().items():
    cell.set_fontsize(10)  # Change to 12 or higher for larger text

# Optionally scale the row/column spacing
table.scale(1.2, 1.4)

# Save as PDF
plt.tight_layout()
plt.savefig("trade_matrix.pdf", bbox_inches="tight")

# Close DB connection
conn.close()
