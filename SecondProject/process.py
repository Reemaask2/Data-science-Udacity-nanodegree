# import libraries
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Define merge_data function
def merge_data(messages, categories):
    """
    Merge the messages and categories DataFrames based on the common column 'id'.
    
    Args:
    - messages (DataFrame): DataFrame containing messages data.
    - categories (DataFrame): DataFrame containing categories data.
    
    Returns:
    - merged_df (DataFrame): DataFrame resulting from the merge operation.
    """
    merged_df = messages.merge(categories, how='inner', on=['id'])
    return merged_df

# Load messages and categories data
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')

# Call the merge_data function
merged_df = merge_data(messages, categories)

    # create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
row = categories.head(1)
    # use this row to extract a list of new column names for categories.
category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    
   # rename the columns of `categories`
categories.columns = category_colnames

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
df.drop('categories', axis=1, inplace=True)

   # concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)


def clean_data(df):
    #dropping duplicates
    df.drop_duplicates(inplace=True)


def save_data(df, database_filepath):
    """Stores df in a SQLite database."""
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('messagescategories', engine, index=False, if_exists='replace')  

  
def main():
    if len(sys.argv) == 4:
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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

if __name__ == '__main__':
    main()