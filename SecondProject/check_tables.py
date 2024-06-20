# check_tables.py
from sqlalchemy import create_engine, inspect

def check_tables(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return tables

if __name__ == '__main__':
    database_filepath = 'DisasterResponse.db'  
    tables = check_tables(database_filepath)
    print("Tables in the database:", tables)