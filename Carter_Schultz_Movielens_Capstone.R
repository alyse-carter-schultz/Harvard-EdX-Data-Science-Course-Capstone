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

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
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

##############################################################
# Methods and Analysis
##############################################################

#this function computes the residual means squared error for a vector of ratings and their corresponding predictors

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))

#average rating:
mu <- mean(edx$rating)
mu  

#predict rating using only the average:
average_only_model <- RMSE(edx$rating, mu)
average_only_model

#add results to model table:
Predictions <- data.frame(model="Average Only Model", RMSE=average_only_model)
Predictions

#Are some movies rated more than others? Graph the discrepancy:
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

#estimate the bias of this predictor(b_i):
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

#graphs the bias in a histogram
b_i %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

#fit prediction model based on movieId field
predicted_ratings <- mu + validation %>% 
  left_join(b_i, by='movieId') %>%
  .$b_i

#add new prediction to table
movie_model_rmse <- RMSE(predicted_ratings, validation$rating)
Predictions <- Predictions %>% add_row(model="Movie-Based Model", RMSE=movie_model_rmse)
Predictions

#Do some users rate more movies than others? graph the discrepancy:
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("User Id")

#computes bias of user averages
b_u <- edx %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#adds user averages to the previous model. 
predicted_ratings <- validation %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(prediction = mu + b_i + b_u) %>%
  .$prediction

#add new model to results table
movie_user_model <- RMSE(predicted_ratings, validation$rating)
Predictions <- Predictions %>% add_row(model="Movie/User Based Model", RMSE=movie_user_model)
Predictions

#Applying Regularization

#define a sequence of lambdas
lambdas <- seq(0, 15, 0.2)

#calculate the predicted ratings using all above values of lambda
rmses <- sapply(lambdas, function(lambda){
  mu <- mean(edx$rating)
  #movie average
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  #user average
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  #predict ratings for validation set
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  #predict RMSE
  return(RMSE(validation$rating, predicted_ratings))
})

#plot lambdas
qplot(lambdas, rmses)  
#lambda which minimizes RMSE
lambdas[which.min(rmses)]
#apply regularized model
regularized_movieIDmodel <- min(rmses)

#add regularized model to results table
Predictions <- Predictions %>% add_row(model="Regularized Movie/User Based Model", RMSE = regularized_movieIDmodel)
Predictions


















