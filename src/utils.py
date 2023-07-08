import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pymongo import MongoClient

from src.exception import CustomException
from src.logger import logging


def whitespace_remover(dataframe):
    # iterating over the columns
    for i in dataframe.columns:
        # checking datatype of each column
        if dataframe[i].dtype == 'object':
            # applying strip function on column
            dataframe[i] = dataframe[i].map(lambda x: x.strip() if isinstance(x, str) else x)
    return dataframe


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def export_collection_as_dataframe(db_name,collection_name):
    try:
        # uniform resource indentifier
        uri ="mongodb+srv://patil:patil@cluster0.zldquum.mongodb.net/?retryWrites=true&w=majority"


        # Create a new client and connect to the server
        mongo_client = MongoClient(uri)

        collection = mongo_client[db_name][collection_name]

        df = pd.DataFrame(list(collection.find()))

        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        df.replace({"na": np.nan}, inplace=True)

        return df

    except Exception as e:
        logging.info('Exception Occured in export_collection function utils')
        raise CustomException(e, sys)

    
   
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')

            report[name] = {'Accuracy': test_model_score, 'Precision': precision, 'Recall': recall, 'F1': f1}

        return report
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
