# Load necessary libraries
library(data.table)
library(dplyr)
library(tidyr)

# Load data
house_dt <- fread("./project/volume/data/raw/Stat_380_housedata.csv")
qc_data <- fread("./project/volume/data/raw/Stat_380_QC_table.csv")
ex_sub <- fread("./project/volume/data/raw/example_sub.csv")

#merge the Qual and Cond variables into the house_dt
house_dt_merged <- house_dt %>%
  left_join(qc_data %>%
              select(qc_code, Qual, Cond), by = c("qc_code"))

# Separate into train and test
train <- house_dt_merged[grepl("^train", house_dt_merged$Id), ]

test <- house_dt_merged[grepl("^test", house_dt_merged$Id), ]

# Order the test data by Id
ordered_test <- test %>%
  mutate(TempId = as.numeric(gsub("test_", "", Id))) %>%  # Extract numeric part from Id
  arrange(TempId) %>%  # Order by the numeric part
  mutate(Id = paste0("test_", row_number()))

# Group the train data by Qual and calculate average SalePrice
avg_price_by_group <- train %>%
  group_by(Qual, Cond) %>%
  summarize(SalePrice = mean(SalePrice, na.rm = TRUE), .groups = "drop")

# Merge the average price table with the ordered test table
test_and_avg_price <- ordered_test %>%
  left_join(avg_price_by_group, by = c("Qual", "Cond"))

# Convert SalePrice to numeric and handle any non-numeric entries
test_and_avg_price2 <- test_and_avg_price %>%
  mutate(SalePrice.y = as.numeric(SalePrice.y)) %>%
  replace_na(list(SalePrice.y = 0)) 

# Rename the SalePrice.y column to SalePrice
final_adjusted <- test_and_avg_price2 %>%
  rename(SalePrice = SalePrice.y)

# Create a new Id column with the format test_1, test_2, ...
final <- final_adjusted %>%
  select(Id, SalePrice)

# Check the mean SalePrice
mean(final$SalePrice)

# Write out the final test table to the process folder as .csv
fwrite(final, "./project/volume/data/processed/final_test_predictions2.csv")

