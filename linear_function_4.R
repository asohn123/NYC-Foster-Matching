install.packages("glmnet")
library(glmnet)

# Function that splits the provided data frame and column into a set of all binary columns
make_column_binary = function(not_discharged, column_name)
{
  column_levels = unique(not_discharged[,column_name])
  for (i in 1:length(column_levels))
  {
    new_column_name = paste(column_name, ".", column_levels[i], sep = "")
    temp_vector = as.factor(as.numeric(not_discharged[,column_name] == column_levels[i]))
    not_discharged[,new_column_name] = temp_vector
  }
  column_num = which(colnames(not_discharged) == column_name)
  return(not_discharged[,-1*column_num])
}

# Function that returns names of all non binary columns in provided data frame
return_non_binary_columns = function(not_discharged)
{
  level_num = list()
  for(i in 1:ncol(not_discharged))
  {
    level_num[i] = length(unique(not_discharged[,i]))
  }
  level_num = as.vector(level_num)
  return(colnames(not_discharged)[which(level_num != 2)])
}

afcars.foster = read.csv("afcars_foster_clean.csv")
afcars.foster[is.na(afcars.foster)] <- 0

not_discharged <- subset(afcars.foster,afcars.foster$DISREASN==0)
not_discharged = not_discharged[,c(7, 43:45, 49:65, 67:74, 104:106)] # Selecting only relevant rows
not_discharged = not_discharged[,c(-22,-24)] # Removing rows with only 1 unique value
non_binary_column_list = return_non_binary_columns(not_discharged) # Finding the names of all non_binary variables
# Creating a data frame that has all binary categorical variables
for(i in 1:length(non_binary_column_list))
{
  not_discharged = make_column_binary(not_discharged, non_binary_column_list[i])
}


for(i in 1:nrow)
  # create train and test sets
  set.seed(1)
  bound1 <- floor((nrow(not_discharged)/4)*3)
  randomized <- not_discharged[sample(nrow(not_discharged)), ]
  train <- randomized[1:bound1, ]
  test <- randomized[(bound1+1):nrow(not_discharged), ]
  
  # create train and test x variable sets from foster family attributes
  x_train_family = subset(train, select=c(7, 43:45, 49:65, 67:74, 104:106))
  x_test_family = subset(test, select=c(7, 43:45, 49:65, 67:74, 104:106))
  x_train_family <- as.matrix(x_train_family)
  x_test_family <- as.matrix(x_test_family)
  
  # create train and test y variable sets from outcomes in SettingLOS column
  y_train = train$SettingLOS
  y_train = as.numeric(unlist(y_train))
  y_test = train$SettingLOS
  y_test = as.numeric(unlist(y_test))

  # set sequence of lambdas we want to test
  grid=10^(-2:10)
  
  # Use 5-fold CV to choose the best value of lambda for lasso regression
  # For the command below, alpha=0: ridge regression, alpha=1: lasso regression
  # Note: to perform normal logistic regression when the response is binary, change "multinomial" to "binomial"
  
  ### LASSO regression ####
  lin_model = glmnet(x_train_family, y_train, alpha = 1)
  
  cv.out = cv.glmnet(x_train_family,y_train, alpha=1, lambda=grid, nfolds=5) 
  
  #plot(cv.out)
  bestlam = cv.out$lambda.min
  
  #Train model with best value of lambda on the training set
  lasso.mod = glmnet(x_train_family, y_train, alpha=1, lambda=bestlam)
  
  #Evaluate this model on the test set
  pred = predict(lasso.mod, x_test_family)
  actual = y_test
  mean((actual-pred)^2) 
  
  # output some info
  lasso.mod
  coef(lasso.mod)
  

