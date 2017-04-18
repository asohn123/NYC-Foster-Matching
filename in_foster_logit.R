install.packages("glmnet")
library(glmnet)


#dataset with  rows of only specific case goals
#find average of SettingLOS
#if above average then 1
#else 0


afcars.foster <- read.csv("afcars_foster_clean.csv", stringsAsFactors = FALSE)

afcars.foster[is.na(afcars.foster)] <- 0
afcars.foster <- subset(afcars.foster,afcars.foster[[67]] == 0) 


count = 1
model_list = list()
for (i in c(1:3,5:6))#iterate through case goals excluding 4 (long term foster care)
{
case_goal_dataset <- subset(afcars.foster,afcars.foster[[46]] == i) 
case_goal_avg <- mean(case_goal_dataset$SettingLOS)

#create new column in case_goal_dataset that is an indicator of whether its over case_goal_avg
#name the new column the case goal number please!
new_column_name = paste(i)
temp_vector = as.numeric(case_goal_dataset$SettingLOS >= case_goal_avg) #SettingLos >= avg? for this case goal?
case_goal_dataset[,new_column_name] = temp_vector


model_list[[count]] = logit2(case_goal_dataset,i)
count = count+1
}


# logistic regression model creation function for children in foster care
logit2 = function(black,case_goal)
{
  # create train and test sets
  set.seed(1)
  bound1 <- floor((nrow(black)/4)*3)
  randomized <- black[sample(nrow(black)), ]
  train <- randomized[1:bound1, ]
  test <- randomized[(bound1+1):nrow(black), ]
  
  # create train and test x variable sets from foster family attributes
  x_train_family = subset(train, select=c(8, 44:46, 50:66, 68:75, 105:107))
  x_test_family = subset(test, select=c(8, 44:46, 50:66, 68:75, 105:107))
  x_train_family <- as.matrix(x_train_family)
  x_test_family <- as.matrix(x_test_family)
  
  # create train and test y variable sets from outcomes in DISREASN column
  
  y_train = train[,toString(case_goal)]
  y_train = as.numeric(unlist(y_train))
  y_train = as.factor(y_train)
  y_test = test[,toString(case_goal)]
  y_test = as.numeric(unlist(y_test))
  y_test = as.factor(y_test)
  
  # set sequence of lambdas we want to test
  grid=10^(-2:10)
  
  # Use 5-fold CV to choose the best value of lambda for lasso regression

  cv.out=cv.glmnet(x_train_family,as.factor(y_train),alpha=1,lambda=grid,family="multinomial",nfolds=5)
  bestlam=cv.out$lambda.min
  bestlam
  
  # Train model with best value of lambda on the training set
  lasso.mod=glmnet(x_train_family, as.factor(y_train), alpha=1, lambda=bestlam, family="multinomial")
  
  # Evaluate this model on the test set
  pred = predict(lasso.mod,x_test_family,type="class");
  table(pred,y_test)
  
  #Error rate
  sum(pred != y_test)/length(pred)
  mean(pred != y_test)
  
  # output some info
  lasso.mod
  coef(lasso.mod)
  
  return(coef(lasso.mod))
}