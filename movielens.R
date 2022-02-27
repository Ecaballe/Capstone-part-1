


#Movielens Data

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)


ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                            title = as.character(title),
                                            genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
      semi_join(edx, by = "movieId") %>%
      semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

############################################
#### End of Initial Code Given
###########################################

###################################################
#### Install any additional libraries to use
###################################################

if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(gtools)) install.packages("gtools", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("ddplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(skimr)) install.packages("skimr", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(ranger)) install.packages("ranger", repos = "http://cran.us.r-project.org")



library(dslabs)
library(broom)
library(gtools)
library(dplyr)
library(ggplot2)
library(lubridate)
library(randomForest)
library(skimr)
library(rpart)
library(ranger)






#################################################################
#### Additional columns added to data to be used in evaluation
################################################################# 


# Create variable to calculate the percent of times a user rates a half point, i.e 2.5, 3.5, 4.5
# value of rate_half is 0 if a whole number, 1 if it has a half increment

edx <- edx %>% mutate(rate_half = (abs(round(rating,0) - rating)*2))

# Group by userId to create user specific data, such as what percentage of time do they rate a half point
# what their average rating and standard deviation are and how many ratings they have down

user_info <- edx %>% group_by(userId) %>% summarize(user_pct_half = mean(rate_half), userAvg = mean(rating), userStd = sd(rating), userReviews = n())
str(user_info)
head(user_info)

#Group by movieId to create movie specific data

movie_info <- edx %>% group_by(movieId) %>% summarize(MovieAvg = mean(rating), MovieStd = sd(rating), MovieReviews = n())
str(movie_info)


#Add Columns to datasets for consideration

edx <- left_join(edx, user_info, by = "userId")
edx <- left_join(edx, movie_info, by = "movieId")

validation <- left_join(validation, user_info, by = "userId")
validation <- left_join(validation, movie_info, by = "movieId")



###########################################
####Reviewing Data
###########################################

# dimensions of edx data
dim(edx)
str(edx)

# dimensions of validation data
dim(validation)

# Summarize the # of users and movies in the data
unique_summary <- edx %>% summarize(unique_users = n_distinct(userId), unique_movies= n_distinct(movieId))
unique_summary

# Create a smaller sample to look at distribution ( Original Dataset was too large to plot distribution)
ind <- edx$userId %>% sample(1000000, replace = FALSE)
edx_dist <- edx[ind,]

#Plot a distribution of ratings
edx_dist %>% 
ggplot(aes(rating)) +
geom_histogram(binwidth = 0.5, fill = "blue", col = "black") +
xlab("Rating Given") +
ggtitle("Distribution of Ratings")

# Quick Analysis of the columns to check for missing data

skimmed <- skim(edx)
skimmed[,1:8]

head(edx,20)

#####################################
# Create a test and train dataset
#####################################

#Split the edx dataset into a 20% test and a 80% train dataset


set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
edxtest_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-edxtest_index,]
edx_temp <- edx[edxtest_index,]

# Make sure userId and movieId in test set are also in train set
edx_test <- edx_temp %>% 
      semi_join(edx_train, by = "movieId") %>%
      semi_join(edx_train, by = "userId")

# Add rows removed from test set back into train set
edx_removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, edx_removed)

rm(edxtest_index, edx_temp, edx_removed)



# Create smaller samples to test code when the whole data set is too large to compute

set.seed(1, sample.kind="Rounding")
ind <- edx_train$userId %>% sample(10000, replace = FALSE)
edx10k <- edx_train[ind,]

set.seed(1, sample.kind="Rounding")
ind <- edx_train$userId %>% sample(100000, replace = FALSE)
edx100k <- edx_train[ind,]

set.seed(1, sample.kind="Rounding")
ind <- edx_train$userId %>% sample(1000000, replace = FALSE)
edx1M <- edx_train[ind,]



#########################################################
### Train the model
#########################################################


#Select method pick predictors (user_pct_half, userAvg, userStd, MovieAvg, MovieStd, MovieReviews)

# Set the size sample for the training data set
train_set <- edx1M

###################knn model with edx10k,edx100k Test_RMSE = 0.8832128, 0.9454087

control <- trainControl(method = "cv")
model_knn <- train(rating ~user_pct_half + userAvg + MovieAvg, data = train_set, method = "knn", tuneGrid = data.frame(k = seq(30, 50, 2)), trControl = control)
model_knn


################## glm model with edx10k,edx100k, edx1M Test_RMSE =0.8714957, 0.8716486, 0.8716314 crashed at 10M

model_glm <- train(rating ~user_pct_half + userAvg + MovieAvg, data = train_set, method = "glm")
model_glm


################### Random Forest model with edx10k, Test_RMSE =0.9597575

control <- trainControl(method = "cv", number = 10)
grid <- data.frame(mtry = c(1, 5, 10))
model_rf <- train(rating ~ user_pct_half + userAvg + MovieAvg, data = train_set, method = "rf", 
	ntree = 20,
	trControl = control,
	tuneGrid = grid,
	nSamp = 5000)


##################### rpart with edx10k, edx100k, edx1M Test_RMSE =  0.8945104, 0.8916313, 0.8918748

#getModelInfo("rpart")
#modelLookup("rpart")


set.seed(1, sample.kind="Rounding")
model_rpart <- train(
  rating ~ user_pct_half + userAvg + MovieAvg, data = train_set, method = "rpart",
  trControl = trainControl("cv", number = 10),
  tuneLength = 20
  )
model_rpart

# Plot model error vs different values of cp (complexity parameter)
plot(model_rpart)

# Print the best tuning parameter cp that minimize the model RMSE
model_rpart$bestTune

# Print the decision tree
plot(model_rpart$finalModel, margin = 0.1)
text(model_rpart$finalModel, cex = 0.75)



################### Ranger with edx10k, Test_RMSE =  0.9072315

model_ranger <- train(rating ~user_pct_half + userAvg + MovieAvg, data = train_set, method = "ranger")
model_ranger

##################################################
######Test the method on the test data
##################################################

#Create function to do a Residual Mean Square Error

RMSE <- function(true_ratings, predicted_ratings){
     sqrt(mean((true_ratings - predicted_ratings)^2))
}


#select the model to use

f_model <-model_rpart

rating_hat <- predict(f_model, edx_test)

Test_RMSE <- RMSE(edx_test$rating, rating_hat)

Test_RMSE



###################################################
############# Final Check model_glm with edx1M = 0.8787814
###################################################

ratingPrediction <- predict(f_model, validation)
Validation_RMSE <- RMSE(validation$rating,ratingPrediction)
Validation_RMSE





