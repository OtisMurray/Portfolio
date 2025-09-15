#features.R

#load libraries
library(data.table)
library(caret)
library(ClusterR)

#set seed
set.seed(3)

#load dataset
data <- fread("project/volume/data/raw/data.csv")

#take id column
id <- data$id
data$id <- NULL

#normalize data
preprocess_params <- preProcess(data, method = c("center", "scale"))
data <- predict(preprocess_params, data)

#save normalized data and preprocessing parameters
save(data, preprocess_params, id, file = "project/volume/data/interim/features.RData")

#do pca and set a variance parameter
pca <- prcomp(data)
variance_explained <- cumsum(pca$sdev^2 / sum(pca$sdev^2))

#set threshold at 95% variance explained
variance_threshold <- 0.95
num_components <- which(variance_explained >= variance_threshold)[1]

#display number of components used for tuning
cat("Number of components retained:", num_components, "\n")

#use the number of components retained
pca_dt <- data.table(unclass(pca)$x[, 1:num_components])

#save PCA data and parameters
save(pca, variance_explained, variance_threshold, num_components, pca_dt, file = "project/volume/data/interim/pca_features.RData")
