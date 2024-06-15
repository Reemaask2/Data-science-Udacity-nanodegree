#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


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

nltk.download(['punkt', 'wordnet'])


# In[24]:


print(f"DATABASE: {'DisasterResponse.db'}")


# In[2]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql ('SELECT * FROM MessagesCategories', engine)
#display (df.head (n=10))
X = df ['message']
y = df.iloc[:,4:]
display (y.head (n=3))


# ### 2. Write a tokenization function to process your text data

# In[3]:


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


# In[4]:


# Here I would like to test the tokenaiztation 
example_text = "This is an example sentence, testing the tokenization."
tokens = tokenize(example_text)
print(tokens)


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[5]:


pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),  # To tokenize text using CountVectorizer method
    ('tfidf', TfidfTransformer()),  # Apply transformation
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classifier using RandomForestClassifier
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# train classifier
pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[8]:


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


# In[9]:


# Make predictions
y_pred = pipeline.predict(X_test)

# Convert predictions to a DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)

# Generate report
report = f1_pre_acc_evaluation(y_test, y_pred_df)
print(report)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[10]:


# Get parameters for the pipeline model
params = pipeline.get_params()


# In[11]:


# Print the parameters
print("Parameters for the pipeline model:")
print(params)


# In[13]:


from sklearn.metrics import make_scorer, accuracy_score
# Define the scorer
scorer = make_scorer(accuracy_score)

# Define the reduced parameters for grid search
parameters = {
    'clf__estimator__n_estimators' : [5]
}

# Create GridSearchCV object
cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, verbose=2, n_jobs=-1)


# In[14]:


# Fit the model
cv.fit(X_train, y_train)


# In[15]:


# Get the best parameters and best score
best_params = cv.best_params_
best_score = cv.best_score_


# In[16]:


print("Best Parameters:")
print(best_params)
print("\nBest Score:")
print(best_score)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[17]:


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


# In[18]:


# Evaluate the model
y_pred = evaluate_model(cv, X_test, y_test)


# In[20]:


from sklearn.metrics import classification_report, f1_score, precision_score, accuracy_score
# Calculate and print overall metrics
overall_report = f1_pre_acc_evaluation(y_test, y_pred)
print("Overall Metrics:")
for metric, score in overall_report.items():
    print(f"{metric}: {score:.4f}")


# Based on shown result above, here is an interpretation of the results:
# - F1 Score: A score of (0.5364) suggests that the balance between precision and recall is moderate but could be improved.
# 
# - Precision: A precision of (0.6679) means that approximately 67% of the model's positive predictions are correct.
# 
# - Accuracy: An accuracy of (0.1806) suggests that the model correctly classifies about 18% of the instances, which is relatively low and indicates room for significant improvement.

# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# ### 9. Export your model as a pickle file

# In[21]:


# Save the best model from GridSearchCV as a pickle file
with open('disaster_response_model.pkl', 'wb') as file:
    pickle.dump(cv.best_estimator_, file)


# ### 10. Use this notebook to complete `train_classifier.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




