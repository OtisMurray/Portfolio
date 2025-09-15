# features.R

#load necessary libraries
library(data.table)
library(xgboost)
library(Metrics)

#load datasets
train_data <- fread("project/volume/data/raw/Stat_380_train.csv")
test_data <- fread("project/volume/data/raw/Stat_380_test.csv")
covar_data <- fread("project/volume/data/raw/covar_data.csv")
example_sub <- fread("project/volume/data/raw/Example_Sub.csv")

#merge covariate data with train and test sets
train_data <- merge(train_data, covar_data, by = "sample_id", all.x = TRUE)
test_data <- merge(test_data, covar_data, by = "sample_id", all.x = TRUE)

#replace missing 'dose_3' with "None", as many are missing dose 3
train_data[, dose_3 := fifelse(is.na(dose_3), "None", dose_3)]
test_data[, dose_3 := fifelse(is.na(dose_3), "None", dose_3)]

#clip negative values to 0 for numerical variables
numerical_vars <- c("days_sinceDose2", "days_sinceDose3", "days_sinceSxLatest")
train_data[, (numerical_vars) := lapply(.SD, pmax, 0), .SDcols = numerical_vars]
test_data[, (numerical_vars) := lapply(.SD, pmax, 0), .SDcols = numerical_vars]

#make encoding for categorical variables
cat_vars <- c("sex", "dose_2", "dose_3", "Sx_severity_most_recent")
for (var in cat_vars) {
  levels_combined <- unique(c(train_data[[var]], test_data[[var]]))
  train_data[, (var) := as.integer(factor(get(var), levels = levels_combined))]
  test_data[, (var) := as.integer(factor(get(var), levels = levels_combined))]
}

#create age groups, under 20, 20-40, 40-60, and over 60
train_data[, age_bin := cut(age, breaks = c(-Inf, 20, 40, 60, Inf), labels = c("under_20", "20_40", "40_60", "over_60"))]
test_data[, age_bin := cut(age, breaks = c(-Inf, 20, 40, 60, Inf), labels = c("under_20", "20_40", "40_60", "over_60"))]
train_data[, age_bin := as.integer(factor(age_bin))]
test_data[, age_bin := as.integer(factor(age_bin))]

#create days_sinceLastDose feature
train_data[, has_dose3 := as.integer(dose_3 != "None")]
test_data[, has_dose3 := as.integer(dose_3 != "None")]
train_data[, days_sinceLastDose := ifelse(has_dose3 == 1, days_sinceDose3, days_sinceDose2)]
test_data[, days_sinceLastDose := ifelse(has_dose3 == 1, days_sinceDose3, days_sinceDose2)]

#recent dose indicator (e.g., last 30 days)
train_data[, recent_dose := as.integer(days_sinceLastDose <= 30)]
test_data[, recent_dose := as.integer(days_sinceLastDose <= 30)]

#interaction terms
train_data[, age_dose2_interaction := age * days_sinceDose2]
train_data[, age_dose3_interaction := age * days_sinceDose3]
test_data[, age_dose2_interaction := age * days_sinceDose2]
test_data[, age_dose3_interaction := age * days_sinceDose3]
train_data[, severity_days_interaction := Sx_severity_most_recent * days_sinceSxLatest]
test_data[, severity_days_interaction := Sx_severity_most_recent * days_sinceSxLatest]

#log transformations of days_since features
train_data[, days_sinceDose2_log := log1p(days_sinceDose2)]
train_data[, days_sinceDose3_log := log1p(days_sinceDose3)]
train_data[, days_sinceSxLatest_log := log1p(days_sinceSxLatest)]
test_data[, days_sinceDose2_log := log1p(days_sinceDose2)]
test_data[, days_sinceDose3_log := log1p(days_sinceDose3)]
test_data[, days_sinceSxLatest_log := log1p(days_sinceSxLatest)]

#principal component analysis on covariates
covar_columns <- paste0("CovVar_", 1:49)
pca_train <- prcomp(train_data[, ..covar_columns], center = TRUE, scale. = TRUE)
pca_test <- predict(pca_train, newdata = test_data[, ..covar_columns])

#select the first 5 principal components
for (i in 1:5) {
  train_data[, paste0("Covar_PC", i) := pca_train$x[, i]]
  test_data[, paste0("Covar_PC", i) := pca_test[, i]]
}

#drop original covariate columns
train_data[, (covar_columns) := NULL]
test_data[, (covar_columns) := NULL]

#severity recency indicator
train_data[, recent_severe_symptom := as.integer(Sx_severity_most_recent >= 3 & days_sinceSxLatest <= 30)]
test_data[, recent_severe_symptom := as.integer(Sx_severity_most_recent >= 3 & days_sinceSxLatest <= 30)]

#update selected features
selected_features <- c(
  "age", "age_bin", "sex", "dose_2", "dose_3",
  "days_sinceDose2", "days_sinceDose3", "days_sinceLastDose",
  "days_sinceSxLatest", "Sx_severity_most_recent",
  "days_sinceDose2_log", "days_sinceDose3_log", "days_sinceSxLatest_log",
  "age_dose2_interaction", "age_dose3_interaction",
  "severity_days_interaction", "recent_dose", "recent_severe_symptom",
  paste0("Covar_PC", 1:5)
)

#create model matrices
x_train <- as.matrix(train_data[, ..selected_features])
x_test <- as.matrix(test_data[, ..selected_features])
y_train <- train_data$ic50_Omicron

#save the processed data
saveRDS(x_train, "project/volume/data/interim/x_train.rds")
saveRDS(x_test, "project/volume/data/interim/x_test.rds")
saveRDS(y_train, "project/volume/data/interim/y_train.rds")
saveRDS(test_data$sample_id, "project/volume/data/interim/sample_id.rds")
