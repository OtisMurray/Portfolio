# model.R

#load processed data
x_train <- readRDS("project/volume/data/interim/x_train.rds")
x_test <- readRDS("project/volume/data/interim/x_test.rds")
y_train <- readRDS("project/volume/data/interim/y_train.rds")
sample_id <- readRDS("project/volume/data/interim/sample_id.rds")

#create DMatrix objects
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test)

#set seed for reproducibility
set.seed(123)

#set XGBoost parameters
xgb_params <- list(
  objective = "reg:squarederror", 
  eval_metric = "rmse",           
  eta = 0.01,                      
  max_depth = 7,                   
  min_child_weight = 6,            
  subsample = 0.8,                 
  colsample_bytree = 0.7,          
  alpha = 3,                       
  lambda = 2,                      
  gamma = 0.45,                    
  booster = "gbtree",              
  tree_method = "hist"
)

#cross-validation
cv_model <- xgb.cv(
  params = xgb_params,
  nfold = 8,
  nrounds = 2000,        
  missing = NA,
  data = dtrain,
  print_every_n = 1,
  early_stopping_rounds = 25
)

#get best number of rounds and test error
best_nrounds <- cv_model$best_iteration
test_error <- cv_model$evaluation_log[best_nrounds, "test_rmse_mean"]
cat("Optimal number of rounds:", best_nrounds, "\n")

#train the final model
final_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = best_nrounds,
  missing = NA
)

#generate predictions
test_predictions <- predict(final_model, dtest)

#create final submission file
final_submission <- data.frame(
  sample_id = sample_id,
  ic50_Omicron = as.numeric(test_predictions)
)

#save the submission file
fwrite(final_submission, "project/volume/data/processed/Final.csv")


#define the logging function
log_file <- "project/src/models/attempts.csv"

log_tuning_attempt <- function(params, best_nrounds, test_error, log_file) {
  #create a data.table with the current attempt's parameters and results
  attempt <- data.table(
    objective = params$objective,
    eval_metric = params$eval_metric,
    eta = params$eta,
    max_depth = params$max_depth,
    min_child_weight = params$min_child_weight,
    subsample = params$subsample,
    colsample_bytree = params$colsample_bytree,
    alpha = params$alpha,
    lambda = params$lambda,
    gamma = params$gamma,
    best_nrounds = best_nrounds,
    test_rmse = test_error
  )
  
  #print params for verification
  print(params)
  
  #append the attempt results to the single CSV log file
  if (!file.exists(log_file)) {
    fwrite(attempt, log_file)
  } else {
    fwrite(attempt, log_file, append = TRUE)
  }
}

#log this attempt's results after cross-validation
log_tuning_attempt(xgb_params, best_nrounds, test_error, log_file)
