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

### SVM models
svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(svmRadial)

## Fit or Tune Model HERE
## Tune smoothness and Laplace here
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 2) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 2, repeats=1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

# Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

  kickedpredictions <- predict(final_wf, new_data=test, type="class")
  
  kaggle_submission <- kickedpredictions %>% 
    bind_cols(., test) %>% 
    rename(IsBadBuy = .pred_class) %>% 
    select(RefId, IsBadBuy)
  
  vroom_write(x = kaggle_submission, file = "./SVMPreds.csv" , delim = ",")
  