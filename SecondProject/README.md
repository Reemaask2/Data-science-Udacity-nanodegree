Disaster Response Pipeline Project

Project Motivation:

In this very insteresting project, I applied my data engineering skills to analyze disaster data from Figure Eight, build a model designed to classify the disaster messages. Through the creation of a machine learning pipeline for categorizing real time messages that were dent during disaster times to direct them to the relevant relief agencies. Moreover, the project has a web application interface, enabling emergency workers to input new messages and promptly receive classification results in multiple categories. This will be shows as  an insightful visualization! :)


File Descriptions:

1. data:

|- disaster_categories.csv # data for processing

|- disaster_messages.csv # data for processing

|- process_data.py # data cleaning pipeline

|- DisasterResponse.db # database to save the clean data

2. models:

|- train_classifier.py # machine learning pipeline

|- classifier.pkl # file allows for the classification of new, unseen messages into predefined categories


3. app:

| - template

| |- master.html # main page of the webpage

| |- go.html # classification result page of the webpage

|- run.py # Flask file to run the webpage


Installatons: 
- NumPy
- Pandas
- Matplotlib
- Plotly
- Sklearn
- Pickle
- Sqlalchemy
- Sqlite3
- nltk
- sys
- nltk
- json

Instructions of how to run:
Run the following commands in the project's root directory to set up both database and model:

- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

  Here is snapshots of the webpage:

![Homepage](https://github.com/Reemaask2/Data-science-Udacity-nanodegree/assets/54121017/d80b84a9-8bbf-4ec8-ae69-96adefbe996b)
![Classification msg](https://github.com/Reemaask2/Data-science-Udacity-nanodegree/assets/54121017/8b523740-eace-44a5-aacd-518466501616)

Down below is the link of the webpage to visit: 
https://dpemaftca3.prod.udacity-student-workspaces.com/ 

Licensing, Authors, Acknowledgements: 

Code templates and data were provided by Udacity, The dataset was originally sourced by Udacity from Figure Eight. Special thanks to both of them!! 

