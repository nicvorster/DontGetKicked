library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(bonsai)
library(lightgbm)
library(discrim)
library(themis)
library(ranger)

train <- vroom("training.csv")
test <- vroom("test.csv")

train$PurchDate <- as.Date(train$PurchDate, format = "%Y-%m-%d")
train$IsBadBuy <- as.factor(train$IsBadBuy)
test$PurchDate <- as.Date(test$PurchDate, format = "%Y-%m-%d")
test$IsBadBuy <- as.factor(test$IsBadBuy)
# Define the preprocessing recipe
  my_recipe <- recipe(IsBadBuy ~ PurchDate + VehYear + VehicleAge + Transmission, data = train) %>%
  step_string2factor(all_nominal()) %>%
  step_impute_median(all_numeric(), -all_outcomes()) %>%
  step_date(PurchDate, features = c("doy", "year")) %>%
  step_rm(PurchDate) %>%  # Removes original PurchDate after processing
    step_unknown(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes())

  ### WORKING RECIPE ###
  my_recipe <- recipe(IsBadBuy ~ PurchDate + VehYear + VehicleAge + Transmission + Make + Nationality, data = train) %>%
    # Convert character columns to factors (excluding the target column)
    step_string2factor(all_nominal(), -all_outcomes()) %>%
    # Handle missing values for predictors only
    step_impute_median(all_numeric(), -all_outcomes()) %>%
    step_impute_mode(all_nominal(), -all_outcomes()) %>%
    # Normalize numeric features
    step_normalize(all_numeric(), -all_outcomes()) %>%
    # Feature engineering for date column
    # step_date(PurchDate, features = c("doy", "month", "year")) %>%
    #  step_rm(PurchDate) %>%  # Remove the original date column
    step_unknown(all_nominal(), -all_outcomes()) %>% 
    # Create dummy variables for categorical features
    step_dummy(all_nominal(), -all_outcomes())
  
  

  prepped_recipe <- prep(my_recipe, training = train)
  transformed_data <- bake(prepped_recipe, new_data = train)
  colnames(transformed_data)  # Should include PurchDate_doy, PurchDate_year
  
  logRegModel <- logistic_reg() %>% #Type of model
    set_engine("glm")
  
  ## Put into a workflow here
  logReg_workflow <- workflow() %>%
    add_recipe(rec) %>%
    add_model(logRegModel) %>%
    fit(data= train)
  ## Make predictions
  kick_predictions <- predict(logReg_workflow,
                                new_data=test,
                                type= "class") # "class" or "prob"
  
  
  kaggle_submission <- kick_predictions %>% 
    bind_cols(., test) %>% 
    rename(IsBadBuy = .pred_class) %>% 
    select(RefId, IsBadBuy)
  
  vroom_write(x = kaggle_submission, file = "./LogPreds.csv" , delim = ",")
  