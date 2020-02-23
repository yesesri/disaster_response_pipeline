# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#========================================================================================================#
def load_data(database_filepath):
    """
        Load data from database

        Arguments:
            db_file_name ->  dataset cleaned by ETL pipeline
            engine       -> database
        Output:
            X and Y data

        """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Figight', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return (X, Y, category_names)
####
#========================================================================================================#
def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    ###
    tokens       = word_tokenize(text)
    lemmatizer   = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return ( clean_tokens )
####
#========================================================================================================#
def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf',MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [10, 20, 40]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return (cv)
####
#========================================================================================================#
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred          = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    Y_predd_df = pd.DataFrame(Y_pred, columns=Y_test.columns)
    for each_col in Y_test.columns:
        print ("\n     >========================================<     \nCategory : %s" % each_col)
        print (classification_report(Y_test[each_col], Y_predd_df[each_col]))
####
#========================================================================================================#
def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
####
#========================================================================================================#
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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
#========================================================================================================#
if __name__ == '__main__':
    main()