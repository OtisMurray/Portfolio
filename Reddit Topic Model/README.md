# Project Description

This project uses R to classify Reddit posts into 11 categories based on text embeddings. It utilizes feature preprocessing, dimensionality reduction, visualization, and XGBoost for classification. It also logs all of the different trials of hyperparameter tuning to keep track of which values for the variables: eta, max_depth, gamma, subsample, colsample_bytree, min_child_weight, lambda, and best_nrounds resulted in the most accurate model predictions.

The Topic Model was then judged on a kaggle competition for how well it was able to predict of an unknown testing dataset.


## Convention

Following this directory structure
```
|--project_name                           <- Project root level that is checked into github
  |--project                              <- Project folder
    |--README.md                          <- Top-level README for developers
    |--volume
    |   |--data
    |   |   |--interim                    <- Intermediate data that has been transformed
    |   |   |--processed                  <- The final model-ready data
    |   |   |--raw                        <- The original data dump
    |   |
    |   |--models                         <- Trained model files that can be read into R or Python
    |
    |--required
    |   |--requirements.txt               <- The required libraries for reproducing the Python environment
    |   |--requirements.r                 <- The required libraries for reproducing the R environment
    |
    |
    |--src
    |   |
    |   |--features                       <- Scripts for turning raw and external data into model-ready data
    |   |   |--build_features.r
    |   |
    |   |--models                         <- Scripts for training and saving models
    |   |   |--train_model.r
    |   |
    |
    |
    |
    |--.getignore                         <- List of files not to sync with github
```

Link for data retrieving data to test model: https://www.kaggle.com/competitions/reddit-topic-model-fl-24/data
