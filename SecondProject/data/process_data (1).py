import pandas as pd
from sqlalchemy import create_engine, inspect

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them.
    
    Args:
    messages_filepath: str. Filepath for the csv file containing messages.
    categories_filepath: str. Filepath for the csv file containing categories.
    
    Returns:
    df: dataframe. A dataframe containing merged content of messages and categories datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    """
    Clean the merged dataframe.
    
    Args:
    df: dataframe. A dataframe containing merged content of messages and categories datasets.
    
    Returns:
    df: dataframe. A cleaned dataframe.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database.
    
    Args:
    df: dataframe. A cleaned dataframe.
    database_filename: str. Filepath for the database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    inspector = inspect(engine)
    
    if 'MessagesCategories' in inspector.get_table_names():
        print('Table already exists. Overwriting the existing table.')
        df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')
    else:
        df.to_sql('MessagesCategories', engine, index=False)
    
    print(f'Data saved to database: {database_filename}')

def main():
    import sys
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()