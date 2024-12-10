library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(bonsai)
library(lightgbm)
library(discrim)
library(themis)
library(ranger)
library(stacks) # you need this library to create a stacked model

train <- vroom("training.csv", na = c("", "NA", "NULL", "NOT AVAIL"))
test <- vroom("test.csv", na = c("", "NA", "NULL", "NOT AVAIL"))


library(ggplot2)
library(corrplot)
library(dplyr)

# Numeric columns from a dataset
numeric_cols <- train %>%
  select(where(is.numeric))

# Compute the correlation matrix
cor_matrix <- cor(numeric_cols, use = "complete.obs")

# Plot the correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         title = "Correlation Plot")



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


############### STACKING MODELS ###################


## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

#### BOOST #####

boost_model <- boost_tree(tree_depth=5,
                          trees=2000,
                          learn_rate=0.005) %>%
  set_engine("lightgbm") %>% 
  set_mode("classification")

boosted_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(boost_model)

boost_models <- fit_resamples(boosted_wf,
                              resamples = folds,
                              metrics = metric_set(roc_auc),
                              control = tunedModel)

##boosted_tuneGrid <- grid_regular(tree_depth(),
  #                               trees(),
   #                              learn_rate(),
    #                             levels = 5)

#boostmod <- boosted_wf %>%
 # tune_grid(resamples = folds,
#            grid = boosted_tuneGrid, 
 #           metrics = metric_set(gain_capture),
  #          control = untunedModel)


######### RANDOM FORREST ########


RF_wf <- workflow() %>%
  add_model(rand_forest(mtry = tune(), min_n = tune(), trees=500) %>%
              set_engine("ranger") %>%
              set_mode("classification")) %>%
  add_recipe(rec)

tuning_grid <- grid_regular(
  mtry(range = c(1, 10)),
  min_n(),
  levels = 5)

rfmodel <- RF_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)


# Specify with models to include
my_stack <- stacks() %>%
  add_candidates(rfmodel) %>%   ###Random Forrest 
  add_candidates(boost_models) #%>%   ##Boosted
 # add_candidates(bayes_mod) ##Naive Bayes

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset

## Use the stacked data to get a prediction
#stack_mod %>% predict(new_data=test)

#stack_preds <- predict(stack_mod, new_data = test) 

#test$MMRCurrentAuctionAveragePrice <- as.character(test$MMRCurrentAuctionAveragePrice)
#test$MMRCurrentAuctionCleanPrice <- as.character(test$MMRCurrentAuctionCleanPrice)
#test$MMRCurrentRetailAveragePrice <- as.character(test$MMRCurrentRetailAveragePrice)
#test$MMRCurrentRetailCleanPrice <- as.character(test$MMRCurrentRetailCleanPrice)

stackpredictions <- predict(stack_mod, new_data=test, type="prob")
  
  kaggle_submission <- stackpredictions %>% 
    bind_cols(., test) %>% 
    rename(IsBadBuy = .pred_1) %>% 
    select(RefId, IsBadBuy)
  
  vroom_write(x = kaggle_submission, file = "./StackPreds.csv" , delim = ",")
  