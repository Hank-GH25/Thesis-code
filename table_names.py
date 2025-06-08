import sqlite3

# Configuration
DATABASE_NAME = "eu_trade.db"  # Change this if your database name is different

def list_tables():
    """Connects to the database and prints all table names."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Query to get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()



    conn.close()

    if tables:
        print("Tables in the database:")
        for table in tables:
            print(f"- {table[0]}")
    else:
        print("No tables found in the database.")

if __name__ == "__main__":
    list_tables()
