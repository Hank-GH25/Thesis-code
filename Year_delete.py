import sqlite3

# Configuration
DATABASE_NAME = "eu_trade.db"  # Change this if your database name is different

# List of tables to delete (cleaned from the provided list)
tables_to_delete = [
    [f"W_{year}" for year in range(2014, 2025)] 
 ] # Adding yearly summary tables

def delete_tables(table_names):
    """Deletes specified tables from the database."""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    try:
        for table in table_names:
            cursor.execute(f'DROP TABLE IF EXISTS "{table}";')
            print(f"Table '{table}' has been deleted.")
        
        conn.commit()
    
    except sqlite3.Error as e:
        print(f"Error deleting tables: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    delete_tables(tables_to_delete)
