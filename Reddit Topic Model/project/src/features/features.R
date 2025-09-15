#features.R

#load necessary packages
library(data.table)
library(dplyr)
library(xgboost)
library(Rtsne)
library(ggplot2)

#read in the data
train_raw <- fread("project/volume/data/raw/kaggle_train.csv")
train_emb <- fread("project/volume/data/raw/train_emb.csv")    
test_emb  <- fread("project/volume/data/raw/test_emb.csv")     
test_raw  <- fread("project/volume/data/raw/kaggle_test.csv")  
example_sub <- fread("project/volume/data/raw/example_sub.csv")

#check if dimensions match
stopifnot(nrow(train_raw) == nrow(train_emb))
stopifnot(ncol(train_emb) == 512)

#define classes in the correct order
classes <- c("cars", "CFB", "Cooking", "MachineLearning", "magicTCG",
             "politics", "RealEstate", "science", "StockMarket", "travel", "videogames")

train_raw$reddit <- factor(train_raw$reddit, levels = classes)

#prepare training matrix and labels for xgboost
X_train <- as.matrix(train_emb)
y_train <- as.numeric(train_raw$reddit) - 1 

#prepare test matrix
X_test <- as.matrix(test_emb)

#PCA (for demonstration; may not be used in final model)
pca <- prcomp(X_train, center = TRUE, scale. = TRUE)

#choose 140 components which were found to represent about 95% of the variance
num_pca_components <- 140
pca_X_train <- pca$x[, 1:num_pca_components]

#project test data using the same rotation/center/scale applied to training
pca_X_test <- predict(pca, newdata = X_test)[, 1:num_pca_components]

#use TSNE for visualization
set.seed(123) 
tsne_out <- Rtsne(pca_X_train, 
                  perplexity = 30, 
                  theta = 0.5, 
                  verbose = TRUE, 
                  max_iter = 500)

#create a data frame for plotting
tsne_results <- data.frame(
  TSNE_1 = tsne_out$Y[,1],
  TSNE_2 = tsne_out$Y[,2],
  Label = train_raw$reddit
)

#plot t-SNE with ggplot2
tsne_plot <- ggplot(tsne_results, aes(x = TSNE_1, y = TSNE_2, color = Label)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "t-SNE Visualization of PCA-Reduced Embeddings", 
       x = "t-SNE Dimension 1", 
       y = "t-SNE Dimension 2")

#print the plot to the console
print(tsne_plot)

#save objects for modeling
saveRDS(list(
  X_train = X_train,
  y_train = y_train,
  X_test = X_test,
  classes = classes,
  test_raw = test_raw,
  pca_X_train = pca_X_train,
  pca_X_test = pca_X_test
), file = "project/volume/data/interim/feature_data.rds")
