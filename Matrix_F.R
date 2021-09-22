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
