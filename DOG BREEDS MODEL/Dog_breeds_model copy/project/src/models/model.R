#model.R

#load saved features
load("project/volume/data/interim/features.RData")
load("project/volume/data/interim/pca_features.RData")

#apply GMM, setting the number of clusters to 4 for the 4 breeds
num_clusters <- 4 
gmm_model <- GMM(pca_dt, num_clusters, dist_mode = "eucl_dist", seed_mode = "random_subset", km_iter = 50, em_iter = 1000, verbose = TRUE)

#save GMM model
save(gmm_model, file = "project/volume/data/interim/gmm_model.RData")

#predict the probabilities for the clusters
prob_cluster <- predict_GMM(
  pca_dt,
  gmm_model$centroids,
  gmm_model$covariance_matrices,
  gmm_model$weights
)

#take the probabilities and assign them to the ids
probabilities <- data.table(prob_cluster$cluster_proba)
probabilities[, id := id]

#switch breeds 3 and 4 so they are assigned the correct cluster
cluster_to_breed <- list(
  "1" = "breed_1",
  "2" = "breed_2",
  "3" = "breed_4", 
  "4" = "breed_3"   
)

#transform the probabilities into the respective predictions
remapped_probabilities <- probabilities[, .(
  id = id,
  breed_1 = .SD[[as.integer(names(cluster_to_breed)[sapply(cluster_to_breed, function(x) x == "breed_1")])]],
  breed_2 = .SD[[as.integer(names(cluster_to_breed)[sapply(cluster_to_breed, function(x) x == "breed_2")])]],
  breed_3 = .SD[[as.integer(names(cluster_to_breed)[sapply(cluster_to_breed, function(x) x == "breed_3")])]],
  breed_4 = .SD[[as.integer(names(cluster_to_breed)[sapply(cluster_to_breed, function(x) x == "breed_4")])]]
), .SDcols = names(probabilities)[1:num_clusters]]

#round the probabilities
remapped_probabilities[, c("breed_1", "breed_2", "breed_3", "breed_4") := lapply(.SD, round, 2), .SDcols = c("breed_1", "breed_2", "breed_3", "breed_4")]

#save final predictions
fwrite(remapped_probabilities, "project/volume/data/processed/predictions.csv")
