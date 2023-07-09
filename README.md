
# Adult Income Price Prediction Project
### This repository contains a project for predicting adult income prices. The goal of this project is to develop a machine learning model that can predict the income level of an individual based on various features such as age, education, occupation, and more.

# Dataset
The project utilizes the Adult Income dataset, which consists of a collection of attributes describing individuals. The dataset contains both categorical and numerical features, making it suitable for training and evaluating machine learning algorithms.

# PROJECT_STRUCTURE:

- artifacts/
  - model.pkl
  - preprocessor.pkl
  - raw.csv
  - test.csv
  - train.csv
-catboost_info
     - learn
     - tmp

- notebook/
    - data/
    - EDA_INCOME.ipynb

- source/
    - components/
        - __init__.py
        - data_ingestion.py
        - data_transformation.py
        - model_trainer.py

    - constant/
        - __init__.py

    - pipeline/
        - __init__.py
        - prediction_pipeline.py
        - training_pipeline.py

    - __init__.py
    - exception.py
    - logger.py
    - utils.py

- static/
    - style.css

- templates/
    - form.html
    - index.html
    - results.html

- .gitignore
- README.md
- app.py
- requirements.txt
- setup.py
- upload_data.py

- mlruns/
    - 0/
        - 4c9d31062ea24e80aeede8209540a279/
            - artifacts/
                - best_model
                - MLmodel
                - conda.yaml
                - model.pkl
                - python_env.yaml
                - requirements.txt
            - metrics/
                - accuracy
                - f1
                - precision
                - recall
            - params/
                - depth
                - l2_leaf_reg
                - learning_rate
            - tags/
                - meta.yaml
