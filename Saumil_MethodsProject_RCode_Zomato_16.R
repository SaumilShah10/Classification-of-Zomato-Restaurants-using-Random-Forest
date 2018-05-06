library(tree)
library(randomForest)
library(caret)
library(xlsx)
library(data.table)
library(rfUtilities)
library(ROCR)
library(e1071)
library(corrplot)

# Loading the dataset
zomato=read.csv("C:\\Users\\ischool-user.LISW16-HDLAP\\Desktop\\zomato.csv")
summary(zomato)

# The average cost of people of all the restaurants mentioned in the dataset is in different
# currencies because these restaurants are of different currencies. Thus, to avoid the change
# of value of money in different currencies we subset the dataset to include only Indian
# restaurants which form almost 90% of the main dataset.

# Making a new dataset of only Indian Restaurants
attach(zomato)
zomato_IndianRestaurants = zomato[Country.Code == 1,]
summary(zomato_IndianRestaurants)
zomato_IndianRestaurants$Has.Table.booking = ifelse(zomato_IndianRestaurants$Has.Table.booking == 'Yes', 1, 0)
zomato_IndianRestaurants$Has.Online.delivery = ifelse(zomato_IndianRestaurants$Has.Online.delivery == 'Yes', 1, 0)


attach(zomato_IndianRestaurants)
dataset = zomato_IndianRestaurants[c(11,13,14,17,20,21)]
summary(dataset)

# Checking the correlation between the different variables
a = zomato_IndianRestaurants[c(11,13,14,17,18,21)]
corx = cor(a)
corx
corrplot::corrplot(cor(a), method = 'circle')

# Dividing the entire dataset into 2 parts: Predictors and Target variable
attach(dataset)
x = dataset[c(1,2,3,4,6)]
y = dataset[,5]

# Checking the distribution of each variable
hist(dataset$Average.Cost.for.two)
hist(dataset$Has.Table.booking)
hist(dataset$Has.Online.delivery)
hist(dataset$Price.range)
hist(dataset$Votes)
# We can see that most of our variables are highly skewed and selecting any parametric method
# would make random assumptions about the distribution of the data and the results can get 
# distorted because of that.

# Training set and Testing set
seed = 7
set.seed(seed)
i = sample(seq(8652), 5768, replace = FALSE)
training_set = dataset[i,]
testing_set = dataset[-i,]

# Decision Tree
set.seed(seed)
tree.rating = tree(Rating.text ~ Price.range + Has.Online.delivery + Votes + Has.Table.booking + Average.Cost.for.two, data = training_set)
plot(tree.rating)
text(tree.rating, pretty = 0)
tree.pred = predict(tree.rating, testing_set, type = "class")

# Calculating the Accuracy, Precision and Recall of the decision tree
Confusion_Matrix_decisiontree = table(tree.pred, testing_set$Rating.text)
Confusion_Matrix_decisiontree
Accuracy_decisiontree = mean(tree.pred==testing_set$Rating.text)
Accuracy_decisiontree
Precision_decisiontree = diag(Confusion_Matrix_decisiontree) / rowSums(Confusion_Matrix_decisiontree)
Precision_decisiontree
Recall_decisiontree = diag(Confusion_Matrix_decisiontree) / colSums(Confusion_Matrix_decisiontree)
Recall_decisiontree
# We cna see that decision tree is only able to classify the restaurants into 3 classes 
# instead of 6. The classes Excellent, Poor and Very good are completely getting ignored.
# Thus we will try and use random forest  because it can handle outliers much better than
# decision tree and it will also help in improving the performance parameters.

# For random forest it is very easy to select a random value of mtry and number of trees but
# that would not give us the best results. Thus, we will tune these parameters to find their 
# values which will give the best results in our purpose.

# Creating a model with default parameters
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric = "Accuracy"
set.seed(seed)
mtry = sqrt(ncol(x))
tunegrid = expand.grid(.mtry = mtry)
rf_default = train(Rating.text~., data = training_set, method = "rf", metric = metric, tuneGrid = tunegrid, trControl = control)
print(rf_default)

# Creating a model with Random Search Mehtod
control = trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(seed)
mtry = sqrt(ncol(x))
mtry
rf_random = train(Rating.text~., data = training_set, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)
# Using this random search method we can see that out of mtry values from 1 to 5 the best 
# value of mtry that will give us the highest accuracy and kappa values is 2. Thus, we 
# select the value of mtry as 2

# Creating a model with Grid Search Method
# Using this method we can consifrm whether our best value of mtry is correct or not
control = trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(seed)
tunegrid = expand.grid(.mtry=c(1:5))
tunegrid
rf_gridsearch = train(Rating.text~., data = training_set, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
# From the results we can see that the best value of mtry is the same i.e. 2

# Algorithm Tune (tuneRF)
set.seed(seed)
bestmtry <- tuneRF(x, y, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)

# Manual Search for finding the best value of number of trees
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(x))))
modellist <- list()
for (ntree in c(1000, 1500, 2000, 2500)) {
  set.seed(seed)
  fit <- train(Rating.text~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results <- resamples(modellist)
summary(results)
dotplot(results)

# Extend Caret
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# train model
control <- trainControl(method="repeatedcv", number=10, repeats=1)
tunegrid <- expand.grid(.mtry=c(1:5), .ntree=c(1000, 1500, 2000))
set.seed(seed)
custom <- train(Rating.text~., data=dataset, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
custom
summary(custom)
plot(custom)
# From this we can compare all the values of mtry and ntrees that give us different values of
# accuracy and kappa. From this we can see the best value of mtry is 2 and the best value of 
# ntree is 1500

# Final Random Forest
require(randomForest)
require(MASS)
set.seed(seed)
rf.rating = randomForest(Rating.text ~ Price.range + Has.Online.delivery + Votes + Has.Table.booking + Average.Cost.for.two, data = training_set, importance = TRUE, ntree = 1500, mtry = 2)
rf.rating
summary(rf.rating)
plot(rf.rating)

# Finding the importance of variables
varImp(rf.rating)
varImpPlot(rf.rating)

# Predictions
RandomForest_Prediction_Trainingset = predict(rf.rating, type = 'response')
RandomForest_Prediction_Trainingset1 = predict(rf.rating, type = 'prob')[,2]
RandomForest_Prediction_Testingset = predict(rf.rating, type = 'response', newdata = testing_set)
RandomForest_Prediction_Testingset1 = predict(rf.rating, type = 'prob', newdata = testing_set)[,2]

# Calculating the Accuracy, Precision and Recall of Random Forest Method
Confusion_Matrix = table(RandomForest_Prediction_Testingset, testing_set$Rating.text)
Confusion_Matrix
Accuracy_randomforest = mean(RandomForest_Prediction_Testingset==testing_set$Rating.text)
Accuracy_randomforest
Precision_randomforest = diag(Confusion_Matrix) / rowSums(Confusion_Matrix)
Precision_randomforest
Recall_randomforest = diag(Confusion_Matrix) / colSums(Confusion_Matrix)
Recall_randomforest
# We can see the improvement in all the performance parameters using Random Forest Method.
