# import libraries
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
import sys
import pickle
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


nltk.download(['punkt', 'wordnet'])
# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table("messagescategories", con=engine)
def tokenize(text):
    
    """
    Function to tokenize text.
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

#Build the model
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),  # To tokenize text using CountVectorizer method
    ('tfidf', TfidfTransformer()),  # Apply transformation
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classifier using RandomForestClassifier
])

#Train the model
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql ('SELECT * FROM messagescategories', engine)
#display (df.head (n=10))
X = df ['message']
y = df.iloc[:,4:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
from sklearn.metrics import classification_report, accuracy_score, precision_score
def f1_pre_acc_evaluation(y_true, y_pred):
    """
    Function to evaluate F1 score, precision, and accuracy.
    """
    # Calculate F1 score, precision, and accuracy
    f1_score = classification_report(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    # Return a formatted report
    report = f"F1 Score:\n{f1_score}\nPrecision: {precision}\nAccuracy: {accuracy}"
    
    return report

# Test the model 
from sklearn.metrics import classification_report

def test_model(y_test, y_pred):
    """
    To print classification reports for each category.
    """
    for column in y_test.columns:
        print(column)
        print(classification_report(y_test[column], y_pred[column]))

# Call the function
test_model(y_test, y_pred)
def save_model(model, model_filepath):
    import pickle

filename = 'model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(cv, file)
    def main():
   if len(sys.argv) == 3:
    database_filepath, model_filepath = sys.argv[1:]
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    #X, Y, category_names = load_data(database_filepath)
    X, Y = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')
else:
    print('Please provide the filepath of the disaster messages database '
          'as the first argument and the filepath of the pickle file to '
          'save the model to as the second argument. \n\nExample: python '
          'train_classifier.py ../data/DisasterResponse.db classifier.pkl')