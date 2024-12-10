library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(bonsai)
library(lightgbm)
library(discrim)
library(themis)

train <- vroom("training.csv")
test <- vroom("test.csv")
view(train)

train$IsBadBuy <- as.factor(train$IsBadBuy)
my_recipe <- recipe(IsBadBuy ~ . , data=train) %>%
  step_impute_knn(MMRAcquisitionAuctionAveragePrice, impute_with = imp_vars(MMRAcquisitionRetailAveragePrice, MMRCurrentAuctionAveragePrice, MMRCurrentRetailAveragePrice), neighbors = 7) %>% 
  step_impute_knn(MMRCurrentAuctionAveragePrice, impute_with = imp_vars(MMRAcquisitionRetailAveragePrice, MMRAcquisitionAuctionAveragePrice, MMRCurrentRetailAveragePrice), neighbors = 7) %>% 
  step_impute_knn(MMRAcquisitionRetailAveragePrice, impute_with = imp_vars(MMRAcquisitionAuctionAveragePrice, MMRCurrentAuctionAveragePrice, MMRCurrentRetailAveragePrice), neighbors = 7) %>% 
  step_impute_knn(MMRCurrentRetailAveragePrice, impute_with = imp_vars(MMRAcquisitionAuctionAveragePrice, MMRCurrentAuctionAveragePrice, MMRAcquisitionRetailAveragePrice), neighbors = 7) %>% 
  step_impute_knn(MMRAcquisitionAuctionCleanPrice	, impute_with = imp_vars(MMRAcquisitonRetailCleanPrice, MMRCurrentAuctionCleanPrice, MMRCurrentRetailCleanPrice), neighbors = 7) %>% 
  step_impute_knn(MMRAcquisitonRetailCleanPrice	, impute_with = imp_vars(MMRAcquisitionAuctionCleanPrice, MMRCurrentAuctionCleanPrice, MMRCurrentRetailCleanPrice), neighbors = 7) %>% 
   step_impute_knn(MMRCurrentAuctionCleanPrice	, impute_with = imp_vars(MMRAcquisitonRetailCleanPrice, MMRAcquisitionAuctionCleanPrice, MMRCurrentRetailCleanPrice), neighbors = 7) %>% 
  step_impute_knn(MMRCurrentRetailCleanPrice	, impute_with = imp_vars(MMRAcquisitonRetailCleanPrice, MMRCurrentAuctionCleanPrice, MMRAcquisitionAuctionCleanPrice), neighbors = 7) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boosted_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

boosted_tuneGrid <- grid_regular(tree_depth(),
                                 trees(),
                                 learn_rate(),
                                 levels = 2)

folds_boost <- vfold_cv(train, v = 2, repeats=1)

cv_results <- boosted_wf %>%
  tune_grid(resamples = folds_boost,
            grid = boosted_tuneGrid, 
            metrics = metric_set(accuracy))

## CV tune, finalize and predict here and save results

bestTune <- cv_results %>%
  select_best()

final_wf <-
  boosted_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

final_wf %>%
  predict(new_data = train, type = "class")

GGG_pred <- predict(final_wf,
                    new_data = test,
                    type = "class")

kaggle_submission <- GGG_pred %>% 
  bind_cols(., test) %>% 
  rename(IsBadBuy = .pred_class) %>% 
  select(RefId, IsBadBuy)

vroom_write(x = kaggle_submission, file = "./BoostedPreds.csv" , delim = ",")



#### BAYES ####

train$IsBadBuy <- as.factor(train$IsBadBuy)

my_recipe <- recipe(IsBadBuy ~ . , data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur
  step_dummy(all_nominal_predictors()) # dummy variable encoding
 # step_normalize(all_predictors()) %>%
 # step_pca(all_predictors(), threshold=0.8) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) # %>%  #target encoding (must
  # also 
 # step_lencode_glm(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>% 
#  step_lencode_bayes(all_nominal_predictors(), outcome = vars(IsBadBuy))


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(smoothness(),
                            Laplace(),
                            levels = 2) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 3, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

test$MMRCurrentAuctionAveragePrice <- as.character(test$MMRCurrentAuctionAveragePrice)

# Prepare the recipe
prepared_recipe <- prep(my_recipe, training = train)

# Apply the recipe to both training and test data
train_processed <- bake(prepared_recipe, new_data = train)
test_processed <- bake(prepared_recipe, new_data = test)


kick_predictions <- predict(final_wf, new_data=test, type="prob")

## Kaggle Submission
kaggle_submission <- kick_predictions %>%
  bind_cols(., test) %>% 
  rename(IsBadBuy= .pred_1) %>% 
  select(RefId, IsBadBuy) 

## Write out the file
vroom_write(x=kaggle_submission, file="./BayesPreds.csv", delim=",")





##### CHATGPT ####

library(recipes)
library(dplyr)

train$PurchDate <- as.Date(train$PurchDate)


# Create a recipe for preprocessing
my_recipe <- recipe(IsBadBuy ~ PurchDate + VehYear + VehicleAge + Transmission + MMRAcquisitionAuctionAveragePrice + MMRAcquisitionAuctionCleanPrice , data = train) %>%
  # Convert relevant character columns to factors
  step_string2factor(all_nominal()) %>%
  # Handle missing values (e.g., imputation using median for numeric features)
  step_impute_median(all_numeric()) %>%
  # Normalize/scale the numeric columns
  step_normalize(all_numeric(), -all_outcomes()) %>%
  # Convert date variable (PurchDate) to a numeric representation (e.g., days since the first purchase)
  step_date(PurchDate, features = "doy") %>%
  # Optionally, you could apply a time-based encoding, but for simplicity, we take the day of the year (yday)
  step_dummy(all_nominal(), -all_outcomes())  # Create dummy variables for categorical features
##step_mutate(across(where(is.character), as.factor))

# Prepare the recipe
prepared_recipe <- prep(my_recipe, training = data)

# Apply the transformations to the dataset
processed_data <- bake(prepared_recipe, new_data = data)

# View the processed data (first few rows)
head(processed_data)

library(randomForest)

# Train a Random Forest model
rf_model <- randomForest(IsBadBuy ~ ., data = processed_data, ntree = 100)

# View the model summary
summary(rf_model)

# Make predictions (on the same data, or you can split into training/testing sets)
predictions <- predict(rf_model, newdata = processed_data)



########


