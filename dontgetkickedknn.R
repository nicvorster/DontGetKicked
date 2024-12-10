library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(bonsai)
library(lightgbm)
library(discrim)
library(themis)
library(ranger)
library(kknn)

train <- vroom("training.csv")
test <- vroom("test.csv")

train$PurchDate <- as.Date(train$PurchDate, format = "%Y-%m-%d")

  
### DR HEATONS RECIPE ###
rec <- recipe(IsBadBuy ~., data = train) %>%
  update_role(RefId, new_role = 'ID') %>%
  update_role_requirements("ID", bake = FALSE) %>%
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_rm(contains('MMR')) %>%
  step_rm(BYRNO, WheelTypeID, VehYear, VNST, VNZIP1, PurchDate, # these variables don't seem very informative, or are repetitive
          AUCGUART, PRIMEUNIT, # these variables have a lot of missing values
          Model, SubModel, Trim) %>% # these variables have a lot of levels - could also try step_other()
  step_corr(all_numeric_predictors(), threshold = .7) %>%
  step_other(all_nominal_predictors(), threshold = .0001) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors())

prep <- prep(rec)
baked <- bake(prep, new_data = train)
###

  
  knn_wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(nearest_neighbor(neighbors = tune(), weight_func = "rectangular") %>%
                set_engine("kknn") %>%
                set_mode("classification"))

  ## Finalize the Workflow & fit it
  final_wf <-
    knn_wf %>%
    fit(data=train)
  
  kick_predictions <- predict(final_wf, new_data=test, type="class")
  
  
  kaggle_submission <- kick_predictions %>% 
    bind_cols(., test) %>% 
    rename(IsBadBuy = .pred_class) %>% 
    select(RefId, IsBadBuy)
  
  vroom_write(x = kaggle_submission, file = "./KNNPreds.csv" , delim = ",")
  
  
  
  ###POTENTIAL???####
  my_recipe <- recipe(IsBadBuy ~ PurchDate + VehicleAge, data = train) %>%
    step_string2factor(all_nominal()) %>%
    step_dummy(all_nominal(), -all_outcomes()) %>%
    step_impute_median(all_numeric(), -all_outcomes()) %>%
    step_normalize(all_numeric(), -all_outcomes())
  
  knn_wf <- workflow() %>%
    add_recipe(my_recipe) %>%
    add_model(nearest_neighbor(neighbors = tune(), weight_func = "rectangular") %>%
                set_engine("kknn") %>%
                set_mode("classification"))
  
  
 