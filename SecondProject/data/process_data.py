import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, on='id', how='inner')
    
    # Create a dataframe of the 36 individual category columns
    categories_split = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories_split.iloc[0]
    
    # Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    
    # Rename the columns of `categories`
    categories_split.columns = category_colnames
    
    # Convert category values to just numbers
    for column in categories_split:
        categories_split[column] = categories_split[column].str[-1]
        categories_split[column] = pd.to_numeric(categories_split[column])
    
    # Drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_split], axis=1)
    
    return df

def clean_data(df):
    # Drop duplicates based on 'message'
    df = df.drop_duplicates(subset=['message'])
    
    # Drop 'child_alone' column
    df = df.drop('child_alone', axis=1, errors='ignore')
    
    # Filter rows where 'related' column is not equal to 2
    df = df[df['related'] != 2]
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')

def main():
    database_filepath = 'DisasterResponse.db'  # Adjust this filepath as needed
    engine = create_engine('sqlite:///' + database_filepath)
    
    try:
        # Check if the table exists in the database
        inspector = inspect(engine)
        if 'MessagesCategories' in inspector.get_table_names():
            # Table exists, read data from it
            df = pd.read_sql('SELECT * FROM MessagesCategories', engine)
            
            # Print the first few rows of the dataframe to verify
            print(df.head())
        else:
            print("Table 'MessagesCategories' does not exist in the database.")
    
    except Exception as e:
        print(f"Error reading data from database: {str(e)}")

if __name__ == '__main__':
    main()