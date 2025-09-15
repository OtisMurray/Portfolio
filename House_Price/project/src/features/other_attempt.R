# Load necessary libraries
library(data.table)
library(dplyr)
library(caret)

# Load data
house_dt <- fread("./project/volume/data/raw/Stat_380_housedata.csv")
qc_data <- fread("./project/volume/data/raw/Stat_380_QC_table.csv")
ex_sub <- fread("./project/volume/data/raw/example_sub.csv")

# Separate into train and test
train <- house_dt[grepl("^train", house_dt$Id), ]
test <- house_dt[grepl("^test", house_dt$Id), ]

# Train linear regression model using building type, lot area, living area, and year built
model <- lm(SalePrice ~ BldgType + LotArea + GrLivArea + YearBuilt, data = train)

# Predict on the test set
test$SalePrice <- predict(model, newdata = test)

#order the tests so they follow the format test_1 test_2 etc
Ordered_test <- test %>%
  mutate(TempId = as.numeric(gsub("test_", "", Id))) %>%  # Extract numeric part from Id
  arrange(TempId) %>%  # Order by the numeric part
  mutate(Id = paste0("test_", row_number()))  # Rename IDs to test_1, test_2, etc.

# Order by the new ID column for final output
final <- Ordered_test %>%
  select(Id, SalePrice)  # Ensure only relevant columns are selected

# Check the mean SalePrice to ensure somewhat close to global average
mean(final$SalePrice)

# Write out the final test table to the process folder as .csv
fwrite(final, "./project/volume/data/processed/final_test_predictions.csv")

