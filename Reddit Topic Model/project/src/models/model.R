#model.R

#load the prepared data from features.R
feature_data <- readRDS("project/volume/data/interim/feature_data.rds")

X_train <- feature_data$X_train
y_train <- feature_data$y_train
X_test  <- feature_data$X_test
classes <- feature_data$classes
test_raw <- feature_data$test_raw

#create DMatrix for training
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

num_class <- length(classes)

#set XGBoost parameters
params <- list(
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = num_class,
  eta = 0.05,
  max_depth = 6,
  gamma = 0,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  lambda = 5
)

#cross-validation to find best iteration
set.seed(123)
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 1
)

best_nrounds <- xgb_cv$best_iteration

#train the final model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)

#predict on test data
pred_probs <- predict(xgb_model, X_test)
pred_matrix <- matrix(pred_probs, ncol = num_class, byrow = TRUE)


#log results for hyperparameter tuning, it is saved in the models folder
run_results <- data.frame(
  timestamp = Sys.time(),
  eta = params$eta,
  max_depth = params$max_depth,
  gamma = params$gamma,
  subsample = params$subsample,
  colsample_bytree = params$colsample_bytree,
  min_child_weight = params$min_child_weight,
  lambda = params$lambda,
  best_nrounds = best_nrounds,
  stringsAsFactors = FALSE
)

#specify the path to the log file
log_path <- "project/src/models/hyperparam_log.csv"

#append new lines of parameters to hyperparameter tuning log
if (!file.exists(log_path)) {
  fwrite(run_results, log_path)
} else {
  fwrite(run_results, log_path, append = TRUE)
}

#create submission
submission <- data.frame(id = test_raw$id)
for (cls in classes) {
  col_name <- paste0("reddit", cls)
  submission[[col_name]] <- pred_matrix[, which(classes == cls)]
}

#look at top of submission
head(submission)

#write out the submission file
fwrite(submission, "project/volume/data/processed/my_submission.csv")
