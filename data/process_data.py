import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load two CSV files (Messages & Categories) , merge them and returned the combined dataframe.
    ------
    Inputs:
    messages_filepath: file path for the messages file
    categories_filepath: file path for the categories file
    
    Outputs:
    df: dataframe of the combined files
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,on=['id'])
    
    return df    


def clean_data(df):
    """
    take dataframe of the combined files and clean it
    ------
    Inputs:
    df: dataframe from the combined files (messages & categories).
    Outputs:
    df: cleaned dataframe 
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories[:-1][:1]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = []

    for i in range(row.shape[1]):
        col = row[i][0]
        col = col[:-2]
        category_colnames.append(col)
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
     
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df, categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # check number of duplicates
    df.duplicated().sum()
    return df

def save_data(df, database_filename):
    """
    store the dataframe into sqlite database.
    ---
    Inputs:
    df: dataframe combined and cleaned
    database_filename: file path of the database file
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('InsertTableName', engine, index=False,if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()