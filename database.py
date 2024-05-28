import sqlite3
import pandas as pd


def create_transactions_table():
    conn = sqlite3.connect('transaction.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            running_balance REAL NOT NULL,
            category TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def save_transaction(df):
    conn = sqlite3.connect('transaction.db')
    c = conn.cursor()

    delete_all_transactions()
    create_transactions_table()

    for row in df.itertuples():
        date_str = row.Date.strftime('%Y-%m-%d')

        if pd.isna(row.Amount):
            continue

        c.execute('''
            INSERT INTO transactions (date, description, amount, running_balance, category)
            VALUES (?, ?, ?, ?, ?)
        ''', (date_str, row.Description, row.Amount, row.Running_Bal, row.Category)
                  )

    conn.commit()
    conn.close()


def delete_all_transactions():
    conn = sqlite3.connect('transaction.db')
    c = conn.cursor()

    c.execute('DROP TABLE transactions')

    conn.commit()
    conn.close()


def load_transactions():
    conn = sqlite3.connect('transaction.db')
    c = conn.cursor()

    c.execute('SELECT date, description, amount, running_balance, category FROM transactions')
    rows = c.fetchall()

    conn.close()

    df = pd.DataFrame(rows, columns=['Date', 'Description', 'Amount', 'Running_Bal', 'Category'])
    df['Date'] = pd.to_datetime(df['Date'])

    return df


def print_table_schema():
    conn = sqlite3.connect('transaction.db')
    c = conn.cursor()

    c.execute('PRAGMA table_info(transactions)')
    schema = c.fetchall()

    conn.close()

    for column in schema:
        print(column)


def print_all_transactions():
    conn = sqlite3.connect('transaction.db')
    c = conn.cursor()

    c.execute('SELECT * FROM transactions')
    transactions = c.fetchall()

    conn.close()

    for transaction in transactions:
        print(transaction)


# create_transactions_table()
# delete_all_transactions()
# df = load_transactions()
# print_all_transactions()
# print_table_schema()
