# features.R

# Load necessary libraries
library(data.table)
library(dplyr)

# Load the datasets
train <- fread("project/volume/data/raw/train_file.csv")
test <- fread("project/volume/data/raw/test_file.csv")
sampleSubmission <- fread("project/volume/data/raw/samp_sub.csv")

# Step 1: Simplify by counting heads and tails and calculating transitions
train_df <- train %>%
  mutate(heads_count = rowSums(select(., starts_with("V")) == 1),  # Count number of heads (1s)
         tails_count = rowSums(select(., starts_with("V")) == 0),  # Count number of tails (0s)
         transitions = sum(abs(diff(as.matrix(select(., starts_with("V")))))),  # Transitions from head to tail or tail to head
         result = result)

test_df <- test %>%
  mutate(heads_count = rowSums(select(., starts_with("V")) == 1),  # Count number of heads (1s)
         tails_count = rowSums(select(., starts_with("V")) == 0),  # Count number of tails (0s)
         transitions = sum(abs(diff(as.matrix(select(., starts_with("V")))))))  # Transitions from head to tail or tail to head

# Step 2: Fit a logistic regression model using the new features
logistic_model <- glm(result ~ heads_count + tails_count + transitions, data = train_df, family = binomial)

# Step 3: Predict the probability of the 11th flip using the new features
test_df$result <- predict(logistic_model, newdata = test_df, type = "response")

# Step 4: Prepare the final submission file
submission <- sampleSubmission
submission$result <- test_df$result

# Step 5: Save the submission file
fwrite(submission, "project/volume/data/processed/submission2.csv", row.names = FALSE)

# Display the first few rows of the final submission for verification
head(submission)


