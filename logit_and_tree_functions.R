# install packages
install.packages("caret")
install.packages("glmnet")
install.packages("tree")
install.packages("MASS")
install.packages("rpart")
library("rpart")

library(caret)
library(glmnet)
library(tree)
library(MASS)

# set working directory
setwd("~/Documents/Columbia/NYC Foster Care Data Challenge/")

# create afcars.foster data frame
afcars.foster <- read.csv("afcars_foster_clean.csv", stringsAsFactors = FALSE)

# turn every NA into a zero
afcars.foster[is.na(afcars.foster)] <- 0

#########################################################
## create new dataframe with child and family clusters ##
#########################################################

# child clusters -> still have lot to discuss on thresholds and splitting
child_age_old <- ifelse(afcars.foster$AgeAtLatRem > 12, 1, 0)
child_female <- ifelse(afcars.foster$SEX == 2, 1, 0)
child_disabled <- ifelse(afcars.foster$CLINDIS == 1, 1, 0)
child_behavior_prob <- ifelse(afcars.foster$CHBEHPRB == 1, 1, 0)
child_neglect <- ifelse(afcars.foster$NEGLECT == 1, 1, 0)
child_moving_high <- ifelse(afcars.foster$NUMPLEP > 2, 1, 0)
child_orig_fam_married <- ifelse(afcars.foster$CTKFAMST == 1, 1, 0)
child_orig_fam_unmarried <- ifelse(afcars.foster$CTKFAMST == 2, 1, 0)
child_orig_fam_sing_female <- ifelse(afcars.foster$CTKFAMST == 3, 1, 0)
child_orig_fam_sing_male <- ifelse(afcars.foster$CTKFAMST == 4, 1, 0)
child_orig_parent_old <- ifelse(2015 - afcars.foster$CTK1YR > 50, 1, 0)
child_in_fc_long <- ifelse(afcars.foster$LifeLOS > 1092, 1, 0)

# family clusters
family_struct_married <- ifelse(afcars.foster$FOSFAMST == 1, 1, 0)
family_struct_unmarried <- ifelse(afcars.foster$FOSFAMST == 2, 1, 0)
family_struct_sing_female <- ifelse(afcars.foster$FOSFAMST == 3, 1, 0)
family_struct_sing_male <- ifelse(afcars.foster$FOSFAMST == 4, 1, 0)
family_metro_million <- ifelse(afcars.foster$RU13 == 1, 1, 0)
family_FIPS_36061 <- ifelse(afcars.foster$FIPSCode == 36061, 1, 0)
family_FIPS_36103 <- ifelse(afcars.foster$FIPSCode == 36103, 1, 0)
family_FIPS_36029 <- ifelse(afcars.foster$FIPSCode == 36029, 1, 0)
family_parent_old <- ifelse(2015 - afcars.foster$FCCTK1YR > 50, 1, 0)
family_payment_high <- ifelse(afcars.foster$FCMntPay > 3817, 1, 0)
family_same_race <- ifelse(afcars.foster$AMIAKN == 1 & afcars.foster$RF1AMAKN == 1, 1,
                           ifelse(afcars.foster$ASIAN == 1 & afcars.foster$RF1ASIAN == 1, 1,
                                  ifelse(afcars.foster$BLKAFRAM == 1 & afcars.foster$RF1BLKAA == 1, 1,
                                         ifelse(afcars.foster$HAWAIIPI == 1 & afcars.foster$RF1NHOPI == 1, 1,
                                                ifelse(afcars.foster$WHITE == 1 & afcars.foster$RF1WHITE == 1, 1,
                                                       ifelse(afcars.foster$UNTODETM == 1 & afcars.foster$RF1UTOD == 1, 1,
                                                              ifelse(afcars.foster$HISORGIN == 1 & afcars.foster$HOFCCTK1 == 1, 1, 0)))))))

# discharge reason outcome column
outcome_discharge_reason <- afcars.foster[, "DISREASN"]

# combine child and family clusters into new dataframe
cluster_df <- data.frame(child_age_old, child_female, child_disabled, child_behavior_prob, child_neglect,
                         child_moving_high, child_orig_fam_married, child_orig_fam_unmarried,
                         child_orig_fam_sing_female, child_orig_fam_sing_male, child_orig_parent_old,
                         child_in_fc_long,
                         family_struct_married, family_struct_unmarried, family_struct_sing_female,
                         family_struct_sing_male, family_metro_million, family_FIPS_36061, family_FIPS_36103,
                         family_FIPS_36029, family_parent_old, family_payment_high, family_same_race,
                         outcome_discharge_reason)

############################################
## creation of logistic regression models ##
############################################

# logistic regression model creation funciton
logit = function(child_cluster_df)
{
  # create train and test sets
  set.seed(1)
  bound1 <- floor((nrow(child_cluster_df)/4)*3)
  randomized <- child_cluster_df[sample(nrow(child_cluster_df)), ]
  train <- randomized[1:bound1, ]
  test <- randomized[(bound1+1):nrow(child_cluster_df), ]
  
  # create train and test x variable sets from foster family attributes
  x_train_family = subset(train, select=c(13:23))
  x_test_family = subset(test, select=c(13:23))
  x_train_family <- as.matrix(x_train_family)
  x_test_family <- as.matrix(x_test_family)
  
  # create train and test y variable sets from outcomes in outcome_discharge_reason column
  y_train = train$outcome_discharge_reason >= 1 & train$outcome_discharge_reason <= 5 & train$outcome_discharge_reason != 4
  y_train = as.numeric(unlist(y_train))
  y_train = as.factor(y_train)
  y_test = test$outcome_discharge_reason >= 1 & test$outcome_discharge_reason <= 5 & test$outcome_discharge_reason != 4
  y_test = as.numeric(unlist(y_test))
  y_test = as.factor(y_test)
  
  # set sequence of lambdas we want to test
  grid=10^(-2:10)
  
  # Use 5-fold CV to choose the best value of lambda for lasso regression
  # For the command below, alpha=0: ridge regression, alpha=1: lasso regression
  # Note: to perform normal logistic regression when the response is binary, change "multinomial" to "binomial"
  cv.out=cv.glmnet(x_train_family,as.factor(y_train),alpha=1,lambda=grid,family="multinomial",nfolds=5)
  bestlam=cv.out$lambda.min
  bestlam
  
  # Train model with best value of lambda on the training set
  lasso.mod=glmnet(x_train_family, as.factor(y_train), alpha=1, lambda=bestlam, family="multinomial")
  
  # Evaluate this model on the test set
  pred = predict(lasso.mod,x_test_family,type="class");
  table(pred,y_test)
  
  #Error rate
  model_error = sum(pred != y_test)/length(pred)
  #ean(pred != y_test)
  
  # output some info
  return_list = list(coef(lasso.mod), model_error)
  names(return_list) = c("model_coef", "model_error")
  
  # return return_list which contains the coefficients of the lasso model and the associated error rate
  return(return_list)
}

################################
## creation of decision trees ##
################################

tree_func = function(child_cluster_df)
{
  # create train and test sets
  set.seed(1)
  bound1 <- floor((nrow(child_cluster_df)/4)*3)
  randomized <- child_cluster_df[sample(nrow(child_cluster_df)), ]
  train <- randomized[1:bound1, ]
  test <- randomized[(bound1+1):nrow(child_cluster_df), ]
  
  # create train and test x variable sets from foster family attributes
  x_train_family = subset(train, select=c(13:23))
  x_test_family = subset(test, select=c(13:23))
  x_train_family <- as.matrix(x_train_family)
  x_test_family <- as.matrix(x_test_family)
  
  # create train and test y variable sets from outcomes in outcome_discharge_reason column
  y_train = train$outcome_discharge_reason >= 1 & train$outcome_discharge_reason <= 5 & train$outcome_discharge_reason != 4
  y_train = as.numeric(unlist(y_train))
  y_train = as.factor(y_train)
  y_test = test$outcome_discharge_reason >= 1 & test$outcome_discharge_reason <= 5 & test$outcome_discharge_reason != 4
  y_test = as.numeric(unlist(y_test))

  temp_train = cbind(y_train,x_train_family)
  temp_train <- data.frame(temp_train)
  temp_train[,1] = as.factor(temp_train[,1])
  foster_tree = tree(y_train ~ ., temp_train)
  cv.foster_tree = cv.tree(foster_tree, FUN=prune.misclass)
  optSize = cv.foster_tree$size[which.min(cv.foster_tree$dev)]
  prune.foster_tree = prune.misclass(foster_tree,best=optSize)
  
  plot(prune.foster_tree)
  text(prune.foster_tree,pretty=0)
  
  temp_test = cbind(test,x_test_family)
  temp_test <- data.frame(temp_test)
  tree.pred = predict(foster_tree, temp_test)
  # tree.pred = predict(prune.foster_tree, temp_test, type = "class")
  model_error = mean(tree.pred!=temp_test$outcome_discharge_reason)
  
  # output some info
  # return_list = list(prune.foster_tree, model_error)
  return_list = list(foster_tree, model_error)
  names(return_list) = c("tree", "model_error")
  
  # return return_list which contains the coefficients of the lasso model and the associated error rate
  return(return_list)
}

###############################################
## create child attribute subset data frames ##
###############################################

HistogramOutput = function(foster_tree, tree_name)
{
  filename_string = paste("./Figures/", tree_name, "Histogram.JPG")
  jpeg(filename = filename_string)
  plot(foster_tree)
  text(foster_tree, pretty = 0)
  dev.off()
  data_out = cbind(names_of_columns, column_num_list, num_na)
  write.csv(data_out, "./Figures/NACounts.csv", row.names = FALSE)
}

###############################################
## create child attribute subset data frames ##
###############################################

# function that runs logit() on each child cluster
run_logit = function(cluster_df)
{
  # set up lists to store child columns and model outputs
  child_columns = c(1:12) # all child variables col nums
  model_list = list()
  error_list = list()
  name_list = list()
  tree_list = list()
  tree_error_list = list()
  
  # loop through child attributes
  for (j in child_columns){
    temp = subset(cluster_df,cluster_df[[j]] == 1)
    temp_model = logit(temp)
    temp_tree = tree_func(temp)
    model_list[j] = temp_model$model_coef
    error_list[j] = temp_model$model_error
    name_list[j] = colnames(cluster_df[j])
    tree_list[j] = temp_tree$tree
    tree_error_list[j] = temp_tree$model_error
  }
  
  # create a list that combines model_list and error_list
  overall_df = cbind(as.vector(name_list), as.vector(error_list), as.vector(model_list), as.vector(tree_error_list), as.vector(tree_list))
  names(overall_df) = c("name_list=", "error_list", "model_list", "tree_error_list", "tree_list")
  
  # return return_list which contains the coefficients of the lasso model and the associated error rate
  return(overall_df)
}

# run the entire script to get list of models with error rates and names
overall_df <- run_logit(cluster_df)
head(overall_df)
