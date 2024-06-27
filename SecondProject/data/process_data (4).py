import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect

def load_data(messages_filepath, categories_filepath):
    """
    Function to load and merge messages and categories datasets
    
    Args:
    messages_filepath: str. Filepath to the messages dataset
    categories_filepath: str. Filepath to the categories dataset
    
    Returns:
    df: pandas DataFrame. Merged DataFrame of messages and categories
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, on='id', how='inner')
    
    return df

def clean_data(df):
    """
    Function to clean the merged DataFrame
    
    Args:
    df: pandas DataFrame. Merged DataFrame of messages and categories
    
    Returns:
    df: pandas DataFrame. Cleaned DataFrame
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    
    # Rename the columns of categories
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    
    # Filter out rows where 'related' is equal to 2
    categories = categories[categories['related'] != 2]
    
    # Drop the original categories column from df
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop 'child_alone' column as it has all zeros
    if 'child_alone' in df.columns:
        df = df.drop('child_alone', axis=1)
    
    return df

def save_data(df, database_filename):
    """
    Function to save the cleaned data to a SQLite database
    
    Args:
    df: pandas DataFrame. Cleaned DataFrame to be saved
    database_filename: str. Filename for the database
    
    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')

def main():
    """
    Main function to execute the ETL pipeline
    
    Args:
    None (expects command line arguments)
    
    Returns:
    None
    """
    # Check for correct number of arguments
    if len(sys.argv) == 4:
        # Assign command line arguments to variables
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths as arguments exactly as shown below:\n\n'
              '    python process_data.py <messages_filepath> <categories_filepath> <database_filepath>\n\n'
              'Arguments:\n'
              '    messages_filepath: Filepath to the CSV file containing messages\n'
              '    categories_filepath: Filepath to the CSV file containing categories\n'
              '    database_filepath: Destination filepath to save the cleaned data in SQLite database')

if __name__ == '__main__':
    main()