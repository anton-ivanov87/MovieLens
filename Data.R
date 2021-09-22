##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

library(tidyverse)
library(caret)
library(data.table)
library(recommenderlab)
library(stringr)
library(lubridate)
library(Matrix)

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
load(movielens.RData)

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
load("data.RData")

# Create a train and test sets

set.seed(2)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index]
test_set <- edx[test_index]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


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

rmse_b_iu <- RMSE(test_set$rating, predicted_ratings_b_u)


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


# Time effect

# Adding year to edx

edx_y <- edx %>%
  mutate (year_m = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(year_m = str_extract(year_m, "\\d{4}")) %>%
  mutate(year_m = as.numeric(year_m),
         year_r = year(as.POSIXct(timestamp, origin="1970-01-01")),
         year_diff = year_r - year_m)

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

results

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

data.frame(lambdas = lambdas1, rmses = rmses_bi) %>%
  ggplot(aes(lambdas, rmses)) +
  geom_point()

lambda1 <- lambdas1[which.min(rmses_bi)]

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

lambdas3 <- seq(20, 30, 0.25)

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

lambdas4 <- seq(0, 100, 10)

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

# Adjusting the predictions

predicted_ratings_reg_corr <- predicted_ratings_reg
predicted_ratings_reg_corr[predicted_ratings_reg_corr > 5] <- 5
predicted_ratings_reg_corr[predicted_ratings_reg_corr < 0.5] <- 0.5

rmse_b_iugy_reg_final <- RMSE(test_set$rating, predicted_ratings_reg_corr)

predicted_ratings_reg_corr2 <- plyr::round_any(predicted_ratings_reg_corr, 0.5)

RMSE(test_set$rating, predicted_ratings_reg_corr2)

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

(1 - rmse_b_iugy_reg/rmse_b_iugy) * 100



# Validation

validation <- validation %>%
  mutate(year_m = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(year_m = str_extract(year_m, "\\d{4}")) %>%
  mutate(year_m = as.numeric(year_m),
         year_r = year(as.POSIXct(timestamp, origin="1970-01-01")),
         year_diff = year_r - year_m)



predicted_ratings_val <-  validation %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_g_reg, by = "genres") %>%
  left_join(b_y_reg, by = "year_diff") %>%
  mutate(pred = mu + b_i_reg + b_u_reg + b_g_reg + b_y_reg) %>%
  pull(pred)

RMSE(validation$rating, predicted_ratings_val)

predicted_ratings_val_corr <- predicted_ratings_val
predicted_ratings_val_corr[predicted_ratings_val_corr > 5] <- 5
predicted_ratings_val_corr[predicted_ratings_val_corr < 0.5] <- 0.5

RMSE(validation$rating, predicted_ratings_val_corr)






# Influence of the time



l3 <- 4

b_g <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  dplyr::summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l3))

predicted_ratings_bg <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)


predicted_ratings_bg_corr <- predicted_ratings_bg
predicted_ratings_bg_corr[predicted_ratings_bg_corr > 5] <- 5
predicted_ratings_bg_corr[predicted_ratings_bg_corr < 0.5] <- 0.5


RMSE(test_set$rating, predicted_ratings_bg)
RMSE(test_set$rating, predicted_ratings_bg_corr)


# Matrix factorization
# Creation of a smaller matrix for speed
train_small <- train_set %>% 
  group_by(movieId) %>%
  filter(n() >= 500) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 500) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

rownames(y)<- y[,1]
y <- y[,-1]

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])

y <- sweep(y, 2, colMeans(y, na.rm=TRUE))
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))

y2 <- as(y, "realRatingMatrix")

svd_pred <- Recommender(data = y2, method = "SVD", parameter = list(k = 30))

svd_pred

sved_res <- predict(object = svd_pred, newdata = y2)
sved_res@items[[1]]



# Adding the year

year <- str_extract(edx$title, "\\(\\d{4}\\)")
year <- str_extract(year, "\\d{4}")
year_r <- as.POSIXct(edx$timestamp, origin="1970-01-01")
year_r <- year(year_r)
edx_y <- edx %>%
  mutate(year_m = as.numeric(year), year_r = year_r, year_diff = year_r - year_m)


table(year_r)


edx_y %>%
  slice(1:10000) %>%
  ggplot(aes(year_diff, rating)) +
  geom_point() +
  geom_smooth(method = "lm")

lambdas4 <- seq(0, 1000, 100)

train_set_y <- train_set %>%
  mutate(year_m = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(year_m = as.numeric(str_extract(year_m, "\\d{4}")),
         year_r = year(as.POSIXct(timestamp, origin="1970-01-01")),
         year_diff = year_r - year_m)

test_set_y <- test_set %>%
  mutate(year_m = str_extract(title, "\\(\\d{4}\\)")) %>%
  mutate(year_m = as.numeric(str_extract(year_m, "\\d{4}")),
         year_r = year(as.POSIXct(timestamp, origin="1970-01-01")),
         year_diff = year_r - year_m)





predicted_ratings_by <- 
  test_set_y %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "year_diff") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y) %>%
  pull(pred)


predicted_ratings_by_corr <- predicted_ratings_by
predicted_ratings_by_corr[predicted_ratings_by_corr > 5] <- 5
predicted_ratings_by_corr[predicted_ratings_by_corr < 0.5] <- 0.5


RMSE(test_set$rating, predicted_ratings_by)
RMSE(test_set$rating, predicted_ratings_by_corr)



# Recommender

train_set_small_cut <- train_set %>%
  select(userId, movieId, rating) %>%
  group_by(userId) %>%
  filter(n() >= 100) %>% ungroup() %>%
  group_by(movieId) %>%
  filter(n() >= 100) %>% ungroup()



test_set_small_cut <- test_set %>%
  select(userId, movieId, rating) %>%
  group_by(userId) %>%
  filter(n() >= 50) %>% ungroup() %>%
  group_by(movieId) %>%
  filter(n() >= 50) %>% ungroup()
  

test_m <- test_set_small_cut %>% 
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

rownames(test_m)<- test_m[,1]
test_m <- test_m[,-1]

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

colnames(test_m) <- with(movie_titles, title[match(colnames(test_m), movieId)])


train_m <- train_set_small_cut %>% 
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

rownames(train_m)<- train_m[,1]
train_m <- test_m[,-1]

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

colnames(train_m) <- with(movie_titles, title[match(colnames(train_m), movieId)])


test_set_small_cut <- as.data.table(test_set_small_cut)

temp_test <- as(test_set_small_cut,"realRatingMatrix")

test_rm <- as(test_m,"realRatingMatrix")
train_rm <- as(train_m,"realRatingMatrix")

test_rm

mod_pop_1 <- Recommender(getData(eval_pop_train, "train"), method = "POPULAR", param = list(normalize = "center"))

pred_pop_1 <- predict(mod_pop_1, getData(eval_pop_test, "known"), type = "ratings")

rmse_pop_1 <- calcPredictionAccuracy(pred_pop_1, getData(eval_pop_test, "unknown"))[1]


calcPredictionAccuracy(pred, test_rm)[1]

eval_pop_test <- evaluationScheme(test_rm, method = "split", train = 0, given = -5)
eval_pop_train <- eval_pop_1


# POPULAR

mod_pop <- Recommender(test_rm, method = "POPULAR", param = list(normalize = "center"))

pred_pop <- predict(mod_pop, test_rm[1:6], type = "ratings")

as(pred_pop, "matrix")[,1:10]

set.seed(3)

eval_pop <- evaluationScheme(test_rm, method = "split", train = 0.7, given = -5)

mod_pop_new <- Recommender(getData(eval_pop, "train"), "POPULAR")

pred_pop_new <- predict(mod_pop, getData(eval_pop, "known"), type = "ratings")

rmse_pop <- calcPredictionAccuracy(pred_pop_new, getData(eval_pop, "unknown"))[1]

# SVD

mod_pop <- Recommender(test_rm, method = "SVD")

pred_pop <- predict(mod_pop, test_rm[1:6], type = "ratings")

as(pred_pop, "matrix")[,1:10]

set.seed(3)

eval <- evaluationScheme(test_rm, method = "split", train = 0.7, given = -5)

mod_svd <- Recommender(getData(eval_svd, "train"), "SVD")

pred_svd <- predict(mod_svd, getData(eval_svd, "known"), type = "ratings")

rmse_svd <- calcPredictionAccuracy(pred_svd, getData(eval_svd, "unknown"))[1]


 
save(temp_test, temp_train, file ="rec_data.RData")
load("rec_data.RData")

save(rec, temp_test, file = "rec_data_2.RData")
load("rec_data_2.RData")



rec <- Recommender(test_rm, method = "SVD")
pre <- predict(rec, test_rm, type = "ratingMatrix")

calcPredictionAccuracy(pre, test_rm) 

RMSE(temp_test@data@x, pre@data@x[1:length(temp_test@data@x)])

head(pre@data@x)

train_small <- train_set %>% 
  group_by(movieId) %>%
  filter(n() >= 500) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 500) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()
rownames(y)<- y[,1]
y <- y[,-1]

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])


z <- test_set %>% 
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

rownames(z)<- z[,1]
z <- z[,-1]

colnames(z) <- with(movie_titles, title[match(colnames(z), movieId)])

rm(rec, temp_test)



# Movielens for Recommenderlab

movielens_small <- movielens %>% 
  group_by(movieId) %>%
  filter(n() >= 300) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 300) %>% ungroup()

movielens_small_m <- movielens_small %>%
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

rownames(movielens_small_m)<- movielens_small_m[,1]
movielens_small_m <- test_m[,-1]

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

colnames(movielens_small_m) <- with(movie_titles, title[match(colnames(movielens_small_m), movieId)])

movielens_small_mr <- as(movielens_small_m,"realRatingMatrix")


# POPULAR

set.seed(3)

eval_pop <- evaluationScheme(movielens_small_mr, method = "split", train = 0.9, given = -5)

mod_pop_new <- Recommender(getData(eval_pop, "train"), "POPULAR")

pred_pop_new <- predict(mod_pop_new, getData(eval_pop, "known"), type = "ratings")

rmse_pop <- calcPredictionAccuracy(pred_pop_new, getData(eval_pop, "unknown"))[1]

# edx for Recommenderlab

edx_small <- edx %>% 
  group_by(movieId) %>%
  filter(n() >= 300) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 300) %>% ungroup()

edx_small_m <- edx_small %>%
  select(userId, movieId, rating) %>%
  pivot_wider(names_from = "movieId", values_from = "rating") %>%
  as.matrix()

rownames(edx_small_m)<- edx_small_m[,1]
edx_small_m <- test_m[,-1]

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

colnames(edx_small_m) <- with(movie_titles, title[match(colnames(edx_small_m), movieId)])

edx_small_mr <- as(edx_small_m,"realRatingMatrix")


# POPULAR

set.seed(3)

eval_pop_edx <- evaluationScheme(edx_small_mr, method = "split", train = 0.8, given = -5)

mod_pop_edx <- Recommender(getData(eval_pop_edx, "train"), "POPULAR")

pred_pop_edx <- predict(mod_pop_edx, getData(eval_pop_edx, "known"), type = "ratings")

rmse_pop_edx <- calcPredictionAccuracy(pred_pop_edx, getData(eval_pop_edx, "unknown"))[1]

test_pred_pop_edx <- as(pred_pop_edx, "matrix")
test_pred_pop_edx[1:10, 1:10]

test_pred_pop_edx_df <- gather(test_pred_pop_edx, "movieId", "rating") 


colnames(test_pred_pop_edx) <- with(movie_titles, title[match(colnames(test_pred_pop_edx), movieId)])
