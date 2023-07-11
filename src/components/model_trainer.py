import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict
from dataclasses import dataclass
import os
import sys
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Naive Bayes": GaussianNB()
            }

            model_report: Dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # Get the best model name
            best_model_name = max(model_report, key=lambda k: model_report[k]['Accuracy'])
            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}')
            print(f'Accuracy Score: {model_report[best_model_name]["Accuracy"]}')
            print('\n====================================================================================\n')
            logging.info(
                f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {model_report[best_model_name]["Accuracy"]}')
            logging.info('Best found model on both training and testing dataset')

            # Perform hyperparameter tuning for the best model
            if best_model_name == "Logistic Regression":
                param_grid = {
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear', 'saga']
                }
            elif best_model_name == "K-Nearest Neighbors":
                param_grid = {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p': [1, 2]
                }
            elif best_model_name == "Decision Tree":
                param_grid = {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [5, 10, 15, 20, None],
                    'max_features': ['sqrt', 'log2', None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif best_model_name == "Random Forest":
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                }
            elif best_model_name == "Gradient Boosting":
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'max_depth': [3, 5, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif best_model_name == "XGBClassifier":
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'max_depth': [3, 5, 10],
                    'subsample': [0.5, 0.8, 1.0],
                    'colsample_bytree': [0.5, 0.8, 1.0],
                    'gamma': [0, 1, 5],
                    'min_child_weight': [1, 5, 10]
                }
            elif best_model_name == "CatBoost Classifier":
                param_grid = {
                    'iterations': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'random_strength': [0, 1, 5, 10],
                    'bagging_temperature': [0.0, 0.5, 1.0],
                    'depth': [3, 5, 10],
                    'l2_leaf_reg': [1, 3, 5],
                    'border_count': [32, 64, 128]
                }
            elif best_model_name == "AdaBoost Classifier":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                }

            grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_model_score = grid_search.best_score_

            print(f'Best Model after Hyperparameter Tuning: {best_model}')
            print(f'Accuracy Score after Hyperparameter Tuning: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(
                f'Best Model after Hyperparameter Tuning: {best_model}, Accuracy Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1= f1_score(y_test, predicted, average='weighted')

            return accuracy, precision, recall, f1_score

        except Exception as e:
            raise CustomException(e, sys)
