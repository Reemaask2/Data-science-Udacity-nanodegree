import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
import sys
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score

nltk.download(['punkt', 'wordnet']))
# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql ('SELECT * FROM MessagesCategories', engine)
#display (df.head (n=10))
X = df ['message']
y = df.iloc[:,4:]
display (y.head (n=3))

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
# Here I would like to test the tokenaiztation 
example_text = "This is an example sentence, testing the tokenization."
tokens = tokenize(example_text)
print(tokens)

# Build the model
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),  # To tokenize text using CountVectorizer method
    ('tfidf', TfidfTransformer()),  # Apply transformation
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classifier using RandomForestClassifier
])
# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train classifier
pipeline.fit(X_train, y_train)

#Test your model
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
# Make predictions
y_pred = pipeline.predict(X_test)

# Convert predictions to a DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)

# Generate report
report = f1_pre_acc_evaluation(y_test, y_pred_df)
print(report)
#Imrpove ypur model
# Get parameters for the pipeline model
params = pipeline.get_params()
# Print the parameters
print("Parameters for the pipeline model:")
print(params)
from sklearn.metrics import make_scorer, accuracy_score
# Define the scorer
scorer = make_scorer(accuracy_score)

# Define the reduced parameters for grid search
parameters = {
    'clf__estimator__n_estimators' : [5]
}

# Create GridSearchCV object
cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, verbose=2, n_jobs=-1)
# Fit the model
cv.fit(X_train, y_train)
# Get the best parameters and best score
best_params = cv.best_params_
best_score = cv.best_score_
print("Best Parameters:")
print(best_params)
print("\nBest Score:")
print(best_score)
#Test your model
# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    for column in y_test.columns:
        print(column)
        print(classification_report(y_test[column], y_pred[:, y_test.columns.get_loc(column)]))
    return y_pred

def f1_pre_acc_evaluation(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    report = {
        'F1 Score': f1,
        'Precision': precision,
        'Accuracy': accuracy
    }
    return report
# Evaluate the model
y_pred = evaluate_model(cv, X_test, y_test)
from sklearn.metrics import classification_report, f1_score, precision_score, accuracy_score
# Calculate and print overall metrics
overall_report = f1_pre_acc_evaluation(y_test, y_pred)
print("Overall Metrics:")
for metric, score in overall_report.items():
    print(f"{metric}: {score:.4f}")
    
# Save the best model from GridSearchCV as a pickle file
with open('disaster_response_model.pkl', 'wb') as file:
    pickle.dump(cv.best_estimator_, file)

