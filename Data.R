##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

library(tidyverse)
library(data.table)
library(caret)
library(stringr)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
save(movielens, file = "movielens.RData")

# A small dataset for knitting
movielens_s <- movielens[1:10,]
save(movielens_s, file = "movielens_s.RData")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
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

save(edx, validation, file ="data.RData")

# Create a train and test sets

set.seed(2)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index]
test_set <- edx[test_index]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

rm(test_index)

#######################################################################
# Create the linear model
#######################################################################

# Loss function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Random prediction

set.seed(3)
pred_random <- sample(seq(0.5, 5, by = 0.5), nrow(test_set), replace = TRUE)

rmse_random <- RMSE(test_set$rating, pred_random)

rm(pred_random)

# Prediction based on the mean rating only

mu <- mean(train_set$rating)

rmse_mean <- RMSE(test_set$rating, mu)


# Prediction based on movie, user, genre and time effects

# Movie effect

b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/n())

predicted_ratings_b_i <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

rmse_b_i <- RMSE(test_set$rating, predicted_ratings_b_i)


# User effect

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/n())

predicted_ratings_b_iu <- test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_b_iu <- RMSE(test_set$rating, predicted_ratings_b_iu)


# Genre effekt

b_g <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/n())

predicted_ratings_b_iug <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

rmse_b_iug <- RMSE(test_set$rating, predicted_ratings_b_iug)

rm(predicted_ratings_b_i, predicted_ratings_b_iu, predicted_ratings_b_iug)

# Time effect

# Adding year to edx

edx <- edx %>%
  mutate (year_m = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(year_m = str_extract(year_m, "\\d{4}")) %>%
  mutate(year_m = as.numeric(year_m),
         year_r = year(as.POSIXct(timestamp, origin="1970-01-01")),
         year_diff = year_r - year_m)

set.seed(4)
edx_y[sample(nrow(edx_y), 100000),] %>%
  ggplot(aes(year_diff, rating)) +
  geom_point()

set.seed(4)
edx_y[sample(nrow(edx_y), 100000),] %>%
  group_by(year_diff) %>%
  summarize(rating_mean = mean(rating)) %>%
  ggplot(aes(year_diff, rating_mean)) +
  geom_point() +
  geom_smooth(method = "lm")

# Adding year to the train and test sets (due to memory problems via mutate, not via join)

train_set <- train_set %>%
  mutate(year_m = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(year_m = str_extract(year_m, "\\d{4}")) %>%
  mutate(year_m = as.numeric(year_m),
         year_r = year(as.POSIXct(timestamp, origin="1970-01-01")),
         year_diff = year_r - year_m)

test_set <- test_set %>%
  mutate(year_m = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(year_m = str_extract(year_m, "\\d{4}")) %>%
  mutate(year_m = as.numeric(year_m),
         year_r = year(as.POSIXct(timestamp, origin="1970-01-01")),
         year_diff = year_r - year_m)

b_y <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  group_by(year_diff) %>%
  summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/n())

predicted_ratings_b_iugy <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "year_diff") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  pull(pred)

rmse_b_iugy <- RMSE(test_set$rating, predicted_ratings_b_iugy)


results <- tibble(method = c("Random", 
                                 "Mean", 
                                 "Movie Effekt", 
                                 "Movie + User Effekt", 
                                 "Movie + User + Genre Effekt",
                                 "Movie + User + Genre + Time Effekt"),
                      RMSE = c(rmse_random,
                               rmse_mean,
                               rmse_b_i,
                               rmse_b_iu,
                               rmse_b_iug,
                               rmse_b_iugy))
results <- results %>%
  mutate(improvement = (1 - RMSE/lag(RMSE))*100)

save(results, file = "results.RData")

rm(b_i, b_u, b_g, b_y, predicted_ratings_b_iugy)

################
# Regularization
################

# Choosing the penalty term for b_i

lambdas1 <- seq(0, 10, 0.25)

rmses_b_i <- sapply(lambdas1, function(l){
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})


# The best lambda shown graphically and as a number

data.frame(lambdas = lambdas1, rmses = rmses_b_i) %>%
  ggplot(aes(lambdas, rmses)) +
  geom_point()

lambda1 <- lambdas1[which.min(rmses_b_i)]

rmse_b_i_reg <- min(rmses_b_i)

b_i_reg <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i_reg = sum(rating - mu)/(n()+lambda1))


# Choosing the penalty term for b_u

lambdas2 <- seq(0, 10, 0.25)

rmses_b_iu <- sapply(lambdas2, function(l){
  
  b_u <- train_set %>% 
    left_join(b_i_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i_reg - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i_reg + b_u) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

# The best lambda shown graphically and as a number

data.frame(lambdas = lambdas2, rmses = rmses_b_iu) %>%
  ggplot(aes(lambdas, rmses)) +
  geom_point()

lambda2 <- lambdas2[which.min(rmses_b_iu)]

rmse_b_iu_reg <- min(rmses_b_iu)

b_u_reg <- train_set %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_reg = sum(rating - b_i_reg - mu)/(n()+lambda2))


# Choosing the penalty term for b_g

lambdas3 <- seq(0, 10, 0.25)

rmses_b_iug <- sapply(lambdas3, function(l){
  
  b_g <- train_set %>% 
    left_join(b_i_reg, by="movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i_reg - b_u_reg - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i_reg + b_u_reg + b_g) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

# The best lambda shown graphically and as a number

data.frame(lambdas = lambdas3, rmses = rmses_b_iug) %>%
  ggplot(aes(lambdas, rmses)) +
  geom_point()

lambda3 <- lambdas3[which.min(rmses_b_iug)]

rmse_b_iug_reg <- min(rmses_b_iug)

b_g_reg <- train_set %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g_reg = sum(rating - b_i_reg - b_u_reg - mu)/(n()+lambda3))


# Choosing the penalty term for b_y

lambdas4 <- seq(300, 400, 10)

rmses_b_iugy <- sapply(lambdas4, function(l){

  b_y <- train_set %>% 
    left_join(b_i_reg, by="movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_g_reg, by = "genres") %>%
    group_by(year_diff) %>%
    summarize(b_y = sum(rating - b_i_reg - b_u_reg - b_g_reg - mu)/(n()+l))
  
  predicted_ratings <-  test_set %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_g_reg, by = "genres") %>%
    left_join(b_y, by = "year_diff") %>%
    mutate(pred = mu + b_i_reg + b_u_reg + b_g_reg + b_y) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
})

# The best lambda shown graphically and as a number

data.frame(lambdas = lambdas4, rmses = rmses_b_iugy) %>%
  ggplot(aes(lambdas, rmses)) +
  geom_point()

lambda4 <- lambdas4[which.min(rmses_b_iugy)]

rmse_b_iugy_reg <- min(rmses_b_iugy)

b_y_reg <- train_set %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_g_reg, by = "genres") %>%
  group_by(year_diff) %>%
  summarize(b_y_reg = sum(rating - b_i_reg - b_u_reg - b_g_reg - mu)/(n()+lambda4))


# Predicted ratings with regularization of all effects

predicted_ratings_reg <-  test_set %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_g_reg, by = "genres") %>%
  left_join(b_y_reg, by = "year_diff") %>%
  mutate(pred = mu + b_i_reg + b_u_reg + b_g_reg + b_y_reg) %>%
  pull(pred)

identical(RMSE(test_set$rating, predicted_ratings_reg), rmse_b_iugy_reg)


# Adjusting the predictions to lie between 0.5 and 5 (stars)

predicted_ratings_reg_corr <- predicted_ratings_reg
predicted_ratings_reg_corr[predicted_ratings_reg_corr > 5] <- 5
predicted_ratings_reg_corr[predicted_ratings_reg_corr < 0.5] <- 0.5

rmse_b_iugy_reg_final <- RMSE(test_set$rating, predicted_ratings_reg_corr)


# Adjusting predictions to have a step 0.5 increases RMSE

predicted_ratings_reg_corr2 <- plyr::round_any(predicted_ratings_reg_corr, 0.5)

RMSE(test_set$rating, predicted_ratings_reg_corr2)

# Summarizing RMSEs

results_reg <- tibble(method = c("Movie Effekt Reg", 
                             "Movie + User Effekt Reg", 
                             "Movie + User + Genre Effekt Reg",
                             "Movie + User + Genre + Time Effekt Reg",
                             "All Reg + Correction"),
                  RMSE = c(rmse_b_i_reg,
                           rmse_b_iu_reg,
                           rmse_b_iug_reg,
                           rmse_b_iugy_reg,
                           rmse_b_iugy_reg_final))

results_reg <- results_reg %>%
  mutate(improvement = (1 - RMSE/lag(RMSE))*100)
  
results <- rbind(results, results_reg)


# Improvement based only on regularization

(1 - rmse_b_iugy_reg/rmse_b_iugy) * 100

rm(b_i_reg, b_u_reg, b_g_reg, b_y_reg, results_reg, lambdas1, lambdas2, lambdas3, lambdas4)

############################################################
# Evaluation of the prediction model with the validation set
############################################################

# Now the effects can be recalculated for the whole edx dataset

mu_edx <- mean(edx$rating)

b_i_edx <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i_edx = sum(rating - mu_edx)/(n()+lambda1))

b_u_edx <- edx %>% 
  left_join(b_i_edx, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_edx = sum(rating - b_i_edx - mu_edx)/(n()+lambda2))

b_g_edx <- edx %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g_edx = sum(rating - b_i_edx - b_u_edx - mu_edx)/(n()+lambda3))

b_y_edx <- edx %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  left_join(b_g_edx, by = "genres") %>%
  group_by(year_diff) %>%
  summarize(b_y_edx = sum(rating - b_i_edx - b_u_edx - b_g_edx - mu_edx)/(n()+lambda4))


# Adding year of movie, year of rating and the difference between them to the validation set

validation <- validation %>%
  mutate(year_m = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(year_m = str_extract(year_m, "\\d{4}")) %>%
  mutate(year_m = as.numeric(year_m),
         year_r = year(as.POSIXct(timestamp, origin="1970-01-01")),
         year_diff = year_r - year_m)

# Final evaluation

predicted_ratings_val <-  validation %>% 
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  left_join(b_g_edx, by = "genres") %>%
  left_join(b_y_edx, by = "year_diff") %>%
  mutate(pred = mu_edx + b_i_edx + b_u_edx + b_g_edx + b_y_edx) %>%
  pull(pred)

predicted_ratings_val[predicted_ratings_val > 5] <- 5
predicted_ratings_val[predicted_ratings_val < 0.5] <- 0.5

rmse_val <- RMSE(validation$rating, predicted_ratings_val)

