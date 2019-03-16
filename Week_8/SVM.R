library(e1071)
library(readr)
library(mosaic)
library(tidyr)


###### For Vowel dataset #####
#Set data up for model
data_setup <- function(data, test_size, target_col){
  data[, -target_col] <- data[, -target_col]
  data_split = {}
  all_rows <- 1:nrow(data)
  test_rows = complete.cases(sample(all_rows, trunc(length(all_rows) * test_size)))
  data_split$Test = data[test_rows,]
  data_split$Train = data[-test_rows,]
  return(data_split)
}

#model for svm
svm_model <- function (vowel, k, g, cost, erroror_change) {
  model <- svm(Class~., data = vowel$Train, kernel = k, gamma = g, cost = cost)
  prediction <- predict(model, vowel$Test)
  confusionMatrix <- table(pred = prediction, true = vowel$Test$Class)
  agreement <- prediction == vowel$Test$Class
  accuracy <- prop.table(table(agreement))
  true_pred <- ifelse(length(table(agreement)) == 1, table(agreement)[1], table(agreement)[2])
  error <- (true_pred - highest_agreemnt) / length(all_rows)
  if (true_pred > highest_agreemnt || (error > erroror_change)) {
    highest_agreemnt <<- true_pred
    best_cost <<- cost
    highest_gamma <<- g
    highest_accuracy <<- accuracy
  }
}


# vowel$Speaker <- as.numeric(vowel$Speaker)
# vowel$Sex <- as.numeric(vowel$Sex)
# My own way of one hot encoding
vowel <- read.csv("vowel.csv", head=TRUE, sep=",", stringsAsFactors=TRUE)
vowel <- vowel %>%
  mutate(value = 1) %>%
  spread(Sex, value,  fill = 0 ) %>%
  mutate(value = 1) %>%
  spread(Speaker, value,  fill = 0)
clean_data <- data_setup(vowel, .3, 11) 
#way to see the highest accuracy you obtained along with the parameters 
#need some sort of way to measure the changes in erroror
#need to initialize what the best is
highest_agreemnt <- 0
highest_gamma <- 0
best_cost <- 0
highest_accuracy <- 0
all_rows <- 1:nrow(vowel)

#different parameters combination set up
for (cost in seq(2^(1:10))) {
  for (gamma in seq(.01, 1, by=.1)) {
    svm_model(clean_data, k = "radial", g = gamma, cost = cost, erroror_change = .1)
  }
}

#model for svm
svm_model <- function (vowel, k, g, cost, erroror_change) {
  model <- svm(Class~., data = vowel$Train, kernel = k, gamma = g, cost = cost)
  prediction <- predict(model, vowel$Test)
  confusionMatrix <- table(pred = prediction, true = vowel$Test$Class)
  agreement <- prediction == vowel$Test$Class
  accuracy <- prop.table(table(agreement))
  true_pred <- ifelse(length(table(agreement)) == 1, table(agreement)[1], table(agreement)[2])
  error <- (true_pred - highest_agreemnt) / length(all_rows)
  if (true_pred > highest_agreemnt || (error > erroror_change)) {
    highest_agreemnt <<- true_pred
    best_cost <<- cost
    highest_gamma <<- g
    highest_accuracy <<- accuracy
  }
}


# Print our results to see
print("Results for Vowels")
print("Agreement")
print(highest_agreemnt)
print("Accuracy")
print(highest_accuracy)
print("Cost")
print(best_cost)
print("Gamma")
print(highest_gamma)




###### For letter dataset #####

###### For letter dataset #####
#Set data up for model
data_setup <- function(data, test_size, target_col){
  data[, -target_col] <- data[, -target_col]
  data_split = {}
  all_rows <- 1:nrow(data)
  test_rows = complete.cases(sample(all_rows, trunc(length(all_rows) * test_size)))
  data_split$Test = data[test_rows,]
  data_split$Train = data[-test_rows,]
  return(data_split)
}

#model for svm
svm_model <- function (letter, k, g, cost, erroror_change) {
  model <- svm(letter ~., data = letter$Train, kernel = k, gamma = g, cost = cost)
  prediction <- predict(model, letter$Test)
  confusionMatrix <- table(pred = prediction, true = letter$Test$letter )
  agreement <- prediction == letter$Test$letter 
  accuracy <- prop.table(table(agreement))
  true_pred <- ifelse(length(table(agreement)) == 1, table(agreement)[1], table(agreement)[2])
  error <- (true_pred - highest_agreemnt) / length(all_rows)
  if (true_pred > highest_agreemnt || (error > erroror_change)) {
    highest_agreemnt <<- true_pred
    best_cost <<- cost
    highest_gamma <<- g
    highest_accuracy <<- accuracy
  }
}



# My own way of one hot encoding
letter <- read.csv("letters.csv", head=TRUE, sep=",", stringsAsFactors=TRUE)
clean_data <- data_setup(letter, .3, 11) 
#way to see the highest accuracy you obtained along with the parameters 
#need some sort of way to measure the changes in erroror
#need to initialize what the best is
highest_agreemnt <- 0
highest_gamma <- 0
best_cost <- 0
highest_accuracy <- 0
all_rows <- 1:nrow(letter)

#different parameters combination set up
for (cost in seq(2^(1:10))) {
  for (gamma in seq(.01, 1, by=.1)) {
    svm_model(clean_data, k = "radial", g = gamma, cost = cost, erroror_change = .1)
  }
}

#model for svm
svm_model <- function (letter, k, g, cost, erroror_change) {
  model <- svm(letter ~., data = letter$Train, kernel = k, gamma = g, cost = cost)
  prediction <- predict(model, letter$Test)
  confusionMatrix <- table(pred = prediction, true = letter$Test$letter )
  agreement <- prediction == letter$Test$letter 
  accuracy <- prop.table(table(agreement))
  true_pred <- ifelse(length(table(agreement)) == 1, table(agreement)[1], table(agreement)[2])
  error <- (true_pred - highest_agreemnt) / length(all_rows)
  if (true_pred > highest_agreemnt || (error > erroror_change)) {
    highest_agreemnt <<- true_pred
    best_cost <<- cost
    highest_gamma <<- g
    highest_accuracy <<- accuracy
  }
}

# Print our results to see
print("Results for Letters")
print("Agreement")
print(highest_agreemnt)
print("Accuracy")
print(highest_accuracy)
print("Cost")
print(best_cost)
print("Gamma")
print(highest_gamma)
