# install packages
install.packages("caret")
install.packages("glmnet")
install.packages("tree")
install.packages("MASS")
install.packages("rpart")
install.packages('slam') 
install.packages('/Library/gurobi701/mac64/R/gurobi_7.0-1.tgz', repos=NULL)
library(rpart)
library(caret)
library(glmnet)
library(tree)
library(MASS)
library(slam)
library(gurobi)

# set working directory
setwd("~/Documents/Columbia/NYC Foster Care Data Challenge/")

# create afcars.foster data frame
afcars.foster <- read.csv("afcars_foster_clean.csv", stringsAsFactors = FALSE)

# turn every NA into a zero
afcars.foster[is.na(afcars.foster)] <- 0

# only NYC (FIPS = 36061)
afcars.foster <- afcars.foster[afcars.foster$FIPSCode == 36061,]

#########################################################
## create new dataframe with child and family clusters ##
#########################################################

# function that clusters original data into intuitive child and family clusters
cluster_func = function(afcars.foster)
{
  # child clusters
  child_preteen <- ifelse(afcars.foster$AgeAtLatRem <= 12, 1, 0)
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
  family_setting_pre_adopt <- ifelse(afcars.foster$CURPLSET == 1, 1, 0)
  family_setting_relative <- ifelse(afcars.foster$CURPLSET == 2, 1, 0)
  family_setting_non_relative <- ifelse(afcars.foster$CURPLSET == 3, 1, 0)
  family_setting_group <- ifelse(afcars.foster$CURPLSET == 4, 1, 0)
  family_setting_inst <- ifelse(afcars.foster$CURPLSET == 5, 1, 0)
  family_setting_indep <- ifelse(afcars.foster$CURPLSET == 6, 1, 0)
  family_setting_runaway <- ifelse(afcars.foster$CURPLSET == 7, 1, 0)
  family_setting_trial <- ifelse(afcars.foster$CURPLSET == 8, 1, 0)
  family_struct_married <- ifelse(afcars.foster$FOSFAMST == 1, 1, 0)
  family_struct_unmarried <- ifelse(afcars.foster$FOSFAMST == 2, 1, 0)
  family_struct_sing_female <- ifelse(afcars.foster$FOSFAMST == 3, 1, 0)
  family_struct_sing_male <- ifelse(afcars.foster$FOSFAMST == 4, 1, 0)
  family_metro_million <- ifelse(afcars.foster$RU13 == 1, 1, 0)
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
  cluster_df <- data.frame(child_preteen, 
                           child_female, child_disabled, child_behavior_prob, child_neglect,
                           child_moving_high, child_orig_fam_married, child_orig_fam_unmarried,
                           child_orig_fam_sing_female, child_orig_fam_sing_male, child_orig_parent_old,
                           child_in_fc_long,
                           family_setting_pre_adopt, family_setting_relative, family_setting_non_relative,
                           family_setting_group, family_setting_inst, family_setting_indep, family_setting_runaway,
                           family_setting_trial, 
                           family_struct_married, family_struct_unmarried, family_struct_sing_female,
                           family_struct_sing_male, family_metro_million, family_parent_old, family_payment_high, family_same_race,
                           outcome_discharge_reason)
  
  # return the new clustered data set
  return(cluster_df)
}

# do the actual clustering
cluster_df <- cluster_func(afcars.foster)

#########################################
## creation of the logistic regression ##
#########################################

# create train and test sets
set.seed(1)
bound1 <- floor((nrow(cluster_df)/4)*3)
randomized <- cluster_df[sample(nrow(cluster_df)), ]
train <- randomized[1:bound1, ]
test <- randomized[(bound1+1):nrow(cluster_df), ]

# create train and test x variable sets from foster family attributes
x_train_family = subset(train, select=c(1:28))
x_test_family = subset(test, select=c(1:28))
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

#########################################
## output from the logistic regression ##
#########################################

# outputs from logistic regression model
summary(lasso.mod)
coef(lasso.mod)
lasso.mod

# Evaluate this model on the test set
pred = predict(lasso.mod,x_test_family,type="class")
table(pred,y_test)

# Error rate
model_error = sum(pred != y_test)/length(pred)
#ean(pred != y_test)

###############################
## create probability matrix ##
###############################

# create the list of 1s and 0s for populating the probability matrix so that every possible combination is reached
# there should be 80 different combinations of 1s and 0s that will be fed into logistic regression model
# for each row to output a final column of success probability
N <- 4
vec <- c(0, 1)
lst <- lapply(numeric(N), function(x) vec)
one_zero <- as.matrix(expand.grid(lst))
one_zero <- data.frame(one_zero)

# assign the ones and zeros to the column attributes
c_preteen <- one_zero$Var1
c_disabled <- one_zero$Var2
c_moving_high <- one_zero$Var3
f_setting_pre_adopt <- c(rep(1,16), rep(0, 64))
f_setting_relative <- c(rep(0,16), rep(1,16), rep(0, 48))
f_setting_non_relative <- c(rep(0,32), rep(1,16), rep(0, 32))
f_setting_group <- c(rep(0,48), rep(1,16), rep(0, 16))
f_setting_inst <- c(rep(0,64), rep(1,16))
f_payment_high <- one_zero$Var4
#f_same_race <- one_zero$Var5

# combine the lists into a data frame
prob_matrix <- data.frame(c_preteen, c_disabled, c_moving_high, f_setting_pre_adopt,
                          f_setting_relative, f_setting_non_relative, f_setting_group,
                          f_setting_inst, f_payment_high) #, f_same_race)

# find means of clusters for attributes not included in the probability matrix
c_female <- list()
c_behavior_prob <- list()
c_neglect <- list()
c_orig_fam_married <- list()
c_orig_fam_unmarried <- list()
c_orig_fam_sing_female <- list()
c_orig_fam_sing_male <- list()
c_orig_parent_old <- list()
c_in_fc_long <- list()
f_setting_indep <- list()
f_setting_runaway <- list()
f_setting_trial <- list()
f_struct_married <- list()
f_struct_unmarried <- list()
f_struct_sing_female <- list()
f_struct_sing_male <- list()
f_metro_million <- list()
f_parent_old <- list()
f_same_race <- list()

# find the means of the centroids for the features of each cluster that are not included in the final matrix
attach(prob_matrix)
for (i in 1:80) {
  c_female[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                            child_disabled == prob_matrix$c_disabled[i] &
                            child_moving_high == prob_matrix$c_moving_high[i] &
                            family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                            family_setting_relative == prob_matrix$f_setting_relative[i] &
                            family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                            family_setting_group == prob_matrix$f_setting_group[i] &
                            family_setting_inst == prob_matrix$f_setting_inst[i] &
                            family_payment_high == prob_matrix$f_payment_high[i])$child_female)
    
  c_behavior_prob[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                   child_disabled == prob_matrix$c_disabled[i] &
                                   child_moving_high == prob_matrix$c_moving_high[i] &
                                   family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                   family_setting_relative == prob_matrix$f_setting_relative[i] &
                                   family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                   family_setting_group == prob_matrix$f_setting_group[i] &
                                   family_setting_inst == prob_matrix$f_setting_inst[i] &
                                   family_payment_high == f_payment_high[i])$child_behavior_prob)
    
  c_neglect[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                             child_disabled == prob_matrix$c_disabled[i] &
                             child_moving_high == prob_matrix$c_moving_high[i] &
                             family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                             family_setting_relative == prob_matrix$f_setting_relative[i] &
                             family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                             family_setting_group == prob_matrix$f_setting_group[i] &
                             family_setting_inst == prob_matrix$f_setting_inst[i] &
                             family_payment_high == prob_matrix$f_payment_high[i])$child_neglect)
    
  c_orig_fam_married[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                      child_disabled == prob_matrix$c_disabled[i] &
                                      child_moving_high == prob_matrix$c_moving_high[i] &
                                      family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                      family_setting_relative == prob_matrix$f_setting_relative[i] &
                                      family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                      family_setting_group == prob_matrix$f_setting_group[i] &
                                      family_setting_inst == prob_matrix$f_setting_inst[i] &
                                      family_payment_high == prob_matrix$f_payment_high[i])$child_orig_fam_married)
    
  c_orig_fam_unmarried[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                        child_disabled == prob_matrix$c_disabled[i] &
                                        child_moving_high == prob_matrix$c_moving_high[i] &
                                        family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                        family_setting_relative == prob_matrix$f_setting_relative[i] &
                                        family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                        family_setting_group == prob_matrix$f_setting_group[i] &
                                        family_setting_inst == prob_matrix$f_setting_inst[i] &
                                        family_payment_high == prob_matrix$f_payment_high[i])$child_orig_fam_unmarried)
    
  c_orig_fam_sing_female[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                          child_disabled == prob_matrix$c_disabled[i] &
                                          child_moving_high == prob_matrix$c_moving_high[i] &
                                          family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                          family_setting_relative == prob_matrix$f_setting_relative[i] &
                                          family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                          family_setting_group == prob_matrix$f_setting_group[i] &
                                          family_setting_inst == prob_matrix$f_setting_inst[i] &
                                          family_payment_high == prob_matrix$f_payment_high[i])$child_orig_fam_sing_female)
    
  c_orig_fam_sing_male[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                        child_disabled == prob_matrix$c_disabled[i] &
                                        child_moving_high == prob_matrix$c_moving_high[i] &
                                        family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                        family_setting_relative == prob_matrix$f_setting_relative[i] &
                                        family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                        family_setting_group == prob_matrix$f_setting_group[i] &
                                        family_setting_inst == prob_matrix$f_setting_inst[i] &
                                        family_payment_high == prob_matrix$f_payment_high[i])$child_orig_fam_sing_male)
    
  c_orig_parent_old[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                     child_disabled == prob_matrix$c_disabled[i] &
                                     child_moving_high == prob_matrix$c_moving_high[i] &
                                     family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                     family_setting_relative == prob_matrix$f_setting_relative[i] &
                                     family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                     family_setting_group == prob_matrix$f_setting_group[i] &
                                     family_setting_inst == prob_matrix$f_setting_inst[i] &
                                     family_payment_high == prob_matrix$f_payment_high[i])$child_orig_parent_old)
    
  c_in_fc_long[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                child_disabled == prob_matrix$c_disabled[i] &
                                child_moving_high == prob_matrix$c_moving_high[i] &
                                family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                family_setting_relative == prob_matrix$f_setting_relative[i] &
                                family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                family_setting_group == prob_matrix$f_setting_group[i] &
                                family_setting_inst == prob_matrix$f_setting_inst[i] &
                                family_payment_high == prob_matrix$f_payment_high[i])$child_in_fc_long)
    
  f_setting_indep[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                   child_disabled == prob_matrix$c_disabled[i] &
                                   child_moving_high == prob_matrix$c_moving_high[i] &
                                   family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                   family_setting_relative == prob_matrix$f_setting_relative[i] &
                                   family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                   family_setting_group == prob_matrix$f_setting_group[i] &
                                   family_setting_inst == prob_matrix$f_setting_inst[i] &
                                   family_payment_high == prob_matrix$f_payment_high[i])$family_setting_indep)
    
  f_setting_runaway[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                     child_disabled == prob_matrix$c_disabled[i] &
                                     child_moving_high == prob_matrix$c_moving_high[i] &
                                     family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                     family_setting_relative == prob_matrix$f_setting_relative[i] &
                                     family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                     family_setting_group == prob_matrix$f_setting_group[i] &
                                     family_setting_inst == prob_matrix$f_setting_inst[i] &
                                     family_payment_high == prob_matrix$f_payment_high[i])$family_setting_runaway)
    
  f_setting_trial[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                   child_disabled == prob_matrix$c_disabled[i] &
                                   child_moving_high == prob_matrix$c_moving_high[i] &
                                   family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                   family_setting_relative == prob_matrix$f_setting_relative[i] &
                                   family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                   family_setting_group == prob_matrix$f_setting_group[i] &
                                   family_setting_inst == prob_matrix$f_setting_inst[i] &
                                   family_payment_high == prob_matrix$f_payment_high[i])$family_setting_trial)
    
  f_struct_married[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                    child_disabled == prob_matrix$c_disabled[i] &
                                    child_moving_high == prob_matrix$c_moving_high[i] &
                                    family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                    family_setting_relative == prob_matrix$f_setting_relative[i] &
                                    family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                    family_setting_group == prob_matrix$f_setting_group[i] &
                                    family_setting_inst == prob_matrix$f_setting_inst[i] &
                                    family_payment_high == prob_matrix$f_payment_high[i])$family_struct_married)
    
  f_struct_unmarried[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                      child_disabled == prob_matrix$c_disabled[i] &
                                      child_moving_high == prob_matrix$c_moving_high[i] &
                                      family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                      family_setting_relative == prob_matrix$f_setting_relative[i] &
                                      family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                      family_setting_group == prob_matrix$f_setting_group[i] &
                                      family_setting_inst == prob_matrix$f_setting_inst[i] &
                                      family_payment_high == prob_matrix$f_payment_high[i])$family_struct_unmarried)
    
  f_struct_sing_female[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                        child_disabled == prob_matrix$c_disabled[i] &
                                        child_moving_high == prob_matrix$c_moving_high[i] &
                                        family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                        family_setting_relative == prob_matrix$f_setting_relative[i] &
                                        family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                        family_setting_group == prob_matrix$f_setting_group[i] &
                                        family_setting_inst == prob_matrix$f_setting_inst[i] &
                                        family_payment_high == prob_matrix$f_payment_high[i])$family_struct_sing_female)
  
  f_struct_sing_male[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                      child_disabled == prob_matrix$c_disabled[i] &
                                      child_moving_high == prob_matrix$c_moving_high[i] &
                                      family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                      family_setting_relative == prob_matrix$f_setting_relative[i] &
                                      family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                      family_setting_group == prob_matrix$f_setting_group[i] &
                                      family_setting_inst == prob_matrix$f_setting_inst[i] &
                                      family_payment_high == prob_matrix$f_payment_high[i])$family_struct_sing_male)
    
  f_metro_million[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                   child_disabled == prob_matrix$c_disabled[i] &
                                   child_moving_high == prob_matrix$c_moving_high[i] &
                                   family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                   family_setting_relative == prob_matrix$f_setting_relative[i] &
                                   family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                   family_setting_group == prob_matrix$f_setting_group[i] &
                                   family_setting_inst == prob_matrix$f_setting_inst[i] &
                                   family_payment_high == prob_matrix$f_payment_high[i])$family_metro_million)
    
  f_parent_old[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                child_disabled == prob_matrix$c_disabled[i] &
                                child_moving_high == prob_matrix$c_moving_high[i] &
                                family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                family_setting_relative == prob_matrix$f_setting_relative[i] &
                                family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                family_setting_group == prob_matrix$f_setting_group[i] &
                                family_setting_inst == prob_matrix$f_setting_inst[i] &
                                family_payment_high == prob_matrix$f_payment_high[i])$family_parent_old)
    
  f_same_race[i] <- mean(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                               child_disabled == prob_matrix$c_disabled[i] &
                               child_moving_high == prob_matrix$c_moving_high[i] &
                               family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                               family_setting_relative == prob_matrix$f_setting_relative[i] &
                               family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                               family_setting_group == prob_matrix$f_setting_group[i] &
                               family_setting_inst == prob_matrix$f_setting_inst[i] &
                               family_payment_high == prob_matrix$f_payment_high[i])$family_same_race)
}
detach(prob_matrix)

# combine the above data into a prediction data frame
prediction_data <- cluster_df[1:80,]
attach(prediction_data)
prediction_data <- cbind(c_preteen, c_female, c_disabled, c_behavior_prob, c_neglect, c_moving_high,
                              c_orig_fam_married, c_orig_fam_unmarried, c_orig_fam_sing_female,
                              c_orig_fam_sing_male, c_orig_parent_old, c_in_fc_long, f_setting_pre_adopt,
                              f_setting_relative, f_setting_non_relative, f_setting_group,
                              f_setting_inst, f_setting_indep, f_setting_runaway, f_setting_trial,
                              f_struct_married, f_struct_unmarried,
                              f_struct_sing_female, f_struct_sing_male, f_metro_million, f_parent_old,
                              f_payment_high, f_same_race, outcome_discharge_reason) #f_same_race, outcome_discharge_reason)
detach(prediction_data)

# output a column of success probability based on the row's criteria
prediction_data <- as.data.frame(prediction_data)
prediction_data[is.na(prediction_data)] <- 0
prediction_data$outcome_discharge_reason <- NULL
prediction_data <- data.matrix(prediction_data)
prob_matrix$success_probability <- 1 - predict(object = lasso.mod, s = bestlam, newx = prediction_data, type = "response")

# output a column of number of cases that fit that row's criteria (size)
for (i in 1:80) {
  prob_matrix$group_size[i] <- nrow(subset(cluster_df, child_preteen == prob_matrix$c_preteen[i] &
                                          child_disabled == prob_matrix$c_disabled[i] &
                                          child_moving_high == prob_matrix$c_moving_high[i] &
                                          family_setting_pre_adopt == prob_matrix$f_setting_pre_adopt[i] &
                                          family_setting_relative == prob_matrix$f_setting_relative[i] &
                                          family_setting_non_relative == prob_matrix$f_setting_non_relative[i] &
                                          family_setting_group == prob_matrix$f_setting_group[i] &
                                          family_setting_inst == prob_matrix$f_setting_inst[i] &
                                          family_payment_high == prob_matrix$f_payment_high[i])) # &
                                         # family_same_race == prob_matrix$f_same_race[i]))
}

# show first few lines of probability matrix
head(prob_matrix)

# create matrix format for viewing and LP
LP_matrix <- matrix(data = NA, nrow = 8, ncol = 10)
for(column in 0:9){
  for(row in 1:8){
    LP_matrix[row, column + 1] <- prob_matrix$success_probability[row + 8 * column]
  }
}

# create group size matrix
size_matrix <- matrix(data = NA, nrow = 8, ncol = 10)
for(column in 0:9){
  for(row in 1:8){
    size_matrix[row, column + 1] <- prob_matrix$group_size[row + 8 * column]
  }
}

# write the prob_matrix to csv
temp1 <- data.frame(prob_matrix)
temp1 <- as.numeric(prob_matrix[,10])
temp1 <- temp1[1:80]
prob_matrix1 <- prob_matrix
prob_matrix1$success_probability <- temp1
write.csv(prob_matrix1, file = "probability_matrix_2.csv")

##############################
## linear programming model ##
##############################
# we want to optimally assign children to families according to our logistic regression model
# and compare our theoretical percentage of children with successful outcomes to the current actual

# child attribute constraints
constraint_vec = c()
for (j in 1:8){
  cmatrix = rep(0,80)
  for (i in 0:9){
    cmatrix[j + 8*i] = 1
  }
  constraint_vec = c(constraint_vec,cmatrix)
}

# family attribute constraints
for (j in 0:9){
  pmatrix = rep(0,80)
  for (i in 1:8){
    pmatrix[i + 8*j] = 1
  }
  constraint_vec = c(constraint_vec,pmatrix)
}

# provide proper inputs for Gurobi
constraint_matrix = matrix(constraint_vec, nrow=18, ncol=80,byrow=T)
model <- list()
model$A <- constraint_matrix
model$obj <- c(LP_matrix)
model$modelsense <- "max"
model$rhs <- c(rowSums(size_matrix),colSums(size_matrix))
model$sense <- rep('=',18)
model$vtype <- 'I'
params <- list(OutputFlag=0)
result <- gurobi(model, params)

# print objective value and allocation solution
optimal_probability <- result$objval
optimal_assignment <- result$x

# compare final theoretical results against the current non-optimal rate of child success
# non-optimally allocated average probability

non_optimal_probability = sum(size_matrix*LP_matrix)/sum(size_matrix)

optimal_probability = optimal_probability/sum(size_matrix)

non_optimal_probability
optimal_probability

percent_change = (optimal_probability - non_optimal_probability)/optimal_probability

percent_change

