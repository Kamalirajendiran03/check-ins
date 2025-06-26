import sqlite3

# Create or connect to a database
conn = sqlite3.connect('test.db')

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Create a simple table for testing
cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_table (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    )
''')

# Insert a test record
cursor.execute('''
    INSERT INTO test_table (name) VALUES ('Test User')
''')

# Commit the changes and close the connection
conn.commit()

# Retrieve data to confirm insertion
cursor.execute('SELECT * FROM test_table')
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
