# import libraries
import pandas as pd
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pickle
import sys


def load_data(database_filepath):
    """
    Load the database table 'InsertTableName' from the provided db file path
    ---
    Inputs:
    database_filepath: database file path
    """
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('InsertTableName',engine)
    X = df['message']
    Y = pd.concat([df[df.columns[4:]] , pd.get_dummies(df['genre'])],axis=1)
    Y = Y.astype(int)
    
    return X,Y,Y.keys()


def tokenize(text):
    """
    process text and return tokens after removing URLs and lemmatization
    ----
    Inputs: 
    text: text/string to be processed
    Outputs:
    clean_tokens: resulted tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    build model through Pipeline
    ---
    Output:
    cv: return the built model
    """
    pipeline = Pipeline([
    ('vector', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('MultiOC', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    print(pipeline.get_params().keys())
    parameters = {#'MultiOC__estimator__max_depth': [10, 20, 30],
              #'MultiOC__estimator__min_samples_leaf':[1,2, 4, 8],
                #'MultiOC__estimator__n_jobs': [1,2,5],
                #'MultiOC__n_jobs': [1,2,5],
                'MultiOC__estimator__n_estimators': [1,5,10]}

    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    run the predict for the model and print classification_report
    ---
    Inputs:
    model: the model to predict
    X_test: X values 
    Y_test: Y values
    category_names: category names for the y values
    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the model to the provided path
    ---
    Inputs:
    model: the model to be saved
    model_filepath: the file path for the saved model
    """
    pickle.dump( model, open( model_filepath, "wb" ) )


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