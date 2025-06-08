import sqlite3
import re

# 🔹 Connect to your SQLite database
db_path = "eu_trade.db"  # ⬅ Change this to your actual database file
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 🔹 Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()  # List of tuples

# 🔹 Loop through tables and rename those with "W_YYYY-MM" format
for (table_name,) in tables:
    match = re.match(r"W_\d{4}-\d{2}", table_name)  # Match "W_YYYY-MM"
    if match:
        new_name = table_name.replace("-", "_")  # Replace hyphen with underscore
        rename_query = f'ALTER TABLE "{table_name}" RENAME TO "{new_name}";'
        cursor.execute(rename_query)
        print(f'Renamed table: {table_name} ➝ {new_name}')

# 🔹 Commit changes and close connection
conn.commit()
conn.close()

print("✅ All tables renamed successfully!")
