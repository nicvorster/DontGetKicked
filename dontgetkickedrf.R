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


### RECIPE ###

rec <- recipe(IsBadBuy ~., data = train) %>%
  update_role(RefId, new_role = 'ID') %>%
  update_role_requirements("ID", bake = FALSE) %>%
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_mutate(VNZIP1 = as.factor(VNZIP1)) %>%
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

#### RF ######
  RF_wf <- workflow() %>%
    add_model(rand_forest(mtry = tune(), min_n = tune(), trees=500) %>%
                set_engine("ranger") %>%
                set_mode("classification")) %>%
    add_recipe(rec)
  
  folds <- vfold_cv(train, v = 5)
  
  tuning_grid <- grid_regular(
    mtry(range = c(1, 10)),
    min_n(),
    levels = 5
  )
  
  CV_results <- RF_wf %>%
    tune_grid(
      resamples = folds,
      grid = tuning_grid,
      metrics = metric_set(roc_auc)
    )

  # Find Best Tuning Parameters
  bestTune <- CV_results %>%
    select_best(metric = "roc_auc")
  
  ## Finalize the Workflow & fit it
  final_wf <-
    RF_wf %>%
    finalize_workflow(bestTune) %>%
    fit(data=train)
  
  test$MMRCurrentAuctionAveragePrice <- as.character(test$MMRCurrentAuctionAveragePrice)
  test$MMRCurrentAuctionCleanPrice <- as.character(test$MMRCurrentAuctionCleanPrice)
  test$MMRCurrentRetailAveragePrice <- as.character(test$MMRCurrentRetailAveragePrice)
  test$MMRCurrentRetailCleanPrice <- as.character(test$MMRCurrentRetailCleanPrice)
  
  #test$PurchDate <- as.character(test$PurchDate)
  
  kickpredictions <- predict(final_wf, new_data=test, type="prob")
  
  kaggle_submission <- kickpredictions %>% 
    bind_cols(., test) %>% 
    rename(IsBadBuy = .pred_1) %>% 
    select(RefId, IsBadBuy)
  
  vroom_write(x = kaggle_submission, file = "./RFPreds.csv" , delim = ",")
  