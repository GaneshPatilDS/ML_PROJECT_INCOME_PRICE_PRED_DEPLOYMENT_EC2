import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from typing import Dict
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient


class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


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
                test_array[:, -1],
            )

            models = {
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Naive Bayes": GaussianNB(),
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )
            print(model_report)
            print("\n====================================================================================\n")
            logging.info(f"Model Report : {model_report}")

            # Find best model without hyperparameter tuning
            best_model_name = max(model_report, key=lambda x: model_report[x]["Accuracy"])
            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name: {best_model_name}")
            print(f"Accuracy Score: {model_report[best_model_name]['Accuracy']}")
            print("\n====================================================================================\n")
            logging.info(
                f"Best Model Found, Model Name: {best_model_name}, Accuracy Score: {model_report[best_model_name]['Accuracy']}"
            )
            logging.info("Best found model on both training and testing dataset")

            # Hyperparameter tuning for the best model
            if best_model_name == "Random Forest":
                param_grid = {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [5, 10, 15, 20, None],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
            elif best_model_name == "XGBClassifier":
                param_grid = {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [5, 10, 15, 20, None],
                    "gamma": [0, 0.5, 1, 1.5, 2],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "colsample_bytree": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "reg_alpha": [0, 0.1, 0.5, 1, 10],
                    "reg_lambda": [0.1, 1, 5, 10, 50, 100],
                }
            elif best_model_name == "CatBoost Classifier":
                param_grid = {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7, 9],
                }
            elif best_model_name == "Gradient Boosting":
                param_grid = {
                    "learning_rate": [0.05, 0.1, 0.2],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 4, 5],
                    "subsample": [0.8, 0.9, 1.0],
                    "max_features": [None, "sqrt", "log2"],
                }
            elif best_model_name == "AdaBoost Classifier":
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "algorithm": ["SAMME", "SAMME.R"],
                }
            elif best_model_name == "Naive Bayes":
                param_grid = {}

            # Perform grid search to find best hyperparameters
            grid_search = GridSearchCV(
                estimator=best_model, param_grid=param_grid, cv=3, scoring="accuracy"
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_model_name = type(best_model).__name__

            print(f"Best Model Found after Hyperparameter Tuning, Model Name: {best_model_name}")
            print(f"Best Model Parameters: {grid_search.best_params_}")
            print("\n====================================================================================\n")
            logging.info(
                f"Best Model Found after Hyperparameter Tuning, Model Name: {best_model_name}"
            )
            logging.info(f"Best Model Parameters: {grid_search.best_params_}")

            mlflow.sklearn.log_model(best_model, "best_model")
            mlflow.log_params(grid_search.best_params_)

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average="weighted")
            recall = recall_score(y_test, predicted, average="weighted")
            f1 = f1_score(y_test, predicted, average="weighted")

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1 Score: {f1}")

            return accuracy, precision, recall, f1

        except Exception as e:
            raise CustomException(e, sys)
