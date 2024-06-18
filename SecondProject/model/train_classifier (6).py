import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Args:
    database_filepath: str. Filepath for the SQLite database.
    
    Returns:
    X: DataFrame. Features dataset.
    Y: DataFrame. Labels dataset.
    category_names: list. List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('MessagesCategories', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and process text data.
    
    Args:
    text: str. Text data.
    
    Returns:
    tokens: list. List of processed tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(tok).strip() for tok in tokens if tok not in stopwords.words("english")]
    return tokens

def build_model():
    """
    Build a machine learning pipeline.
    
    Returns:
    model: Pipeline. Machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance.
    
    Args:
    model: Pipeline. Trained machine learning model.
    X_test: DataFrame. Test features.
    Y_test: DataFrame. Test labels.
    category_names: list. List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f'Category: {col}\n', classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file.
    
    Args:
    model: Pipeline. Trained machine learning model.
    model_filepath: str. Filepath for the pickle file.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()