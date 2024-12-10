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

rec <- recipe(IsBadBuy ~., data = train) %>%
  update_role(RefId, new_role = 'ID') %>%
  update_role_requirements("ID", bake = FALSE) %>%
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
 # step_rm(contains('MMR')) %>%
  step_rm(BYRNO, WheelTypeID, VehYear, VNST, VNZIP1, PurchDate, # these variables don't seem very informative, or are repetitive
          AUCGUART, PRIMEUNIT, # these variables have a lot of missing values
          Model, SubModel, Trim) %>% # these variables have a lot of levels - could also try step_other()
  step_corr(all_numeric_predictors(), threshold = .7) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors())

prep <- prep(rec)
baked <- bake(prep, new_data = train)

## BART MODEL

bart_mod <- parsnip::bart(
  mode = "classification",
  engine = "dbarts",
  trees = 100,
  prior_terminal_node_coef = NULL,
  prior_terminal_node_expo = NULL,
  prior_outcome_range = NULL
)


bart_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(bart_mod) %>% 
  fit(data = train)


## Finalize the Workflow & fit it
final_wf <-
  bart_wf %>%
  fit(data=train)
  
test$MMRCurrentAuctionAveragePrice <- as.character(test$MMRCurrentAuctionAveragePrice)
test$MMRCurrentAuctionCleanPrice <- as.character(test$MMRCurrentAuctionCleanPrice)
test$MMRCurrentRetailAveragePrice <- as.character(test$MMRCurrentRetailAveragePrice)
test$MMRCurrentRetailCleanPrice <- as.character(test$MMRCurrentRetailCleanPrice)

  kickedpredictions <- predict(final_wf, new_data=test, type="prob")
  
  kaggle_submission <- kickedpredictions %>% 
    bind_cols(., test) %>% 
    rename(IsBadBuy = .pred_1) %>% 
    select(RefId, IsBadBuy)
  
  vroom_write(x = kaggle_submission, file = "./BartPreds.csv" , delim = ",")
  