---
title: "Auto Data Analysis: Predicting Car Mileage"
author: "Phaneesha Chilaveni"
date: "11/7/2021"
output: html_document
---



#Load the all the required libraries for the assignment
 
library(ISLR2)
library(ISLR)
library(bootstrap)
library(boot)
library(leaps)
library(klaR)
library(class)
library(GGally)
library(corrplot)
library(caret)
library("rpart")
  

##**In this problem, you will develop a model to predict whether given car gets high or low gas mileage based on the Auto data set.**##

#_Load the data_
 
data(Auto)
auto = Auto
summary(auto)
str(auto)
  

##** (a) Create a binary variable, mpg01, that contains a 1 if mpg contains a value above its median, and a 0 if mpg contains a value below its median. You can compute the median using the median() function. Note you may find it helpful to use the data.frame() function to create a single data set containing both mpg01 and the other Auto variables.**##

#_Creating the binary variable mpg01 and adding it to the original data_
 
mpg01 = rep(0, length(auto$mpg))
mpg01[auto$mpg > median(auto$mpg)] = 1
auto = data.frame(auto[,-1], mpg01)
  


##**(b) Explore the data graphically in order to investigate the association between mpg01 and the other features. Which of the other features seem most likely to be useful in predicting mpg01? Scatterplots and boxplots may be useful tools to answer this question. Describe your findings.**##

#_Using corrplot to see any correlation between mpg01 and other parameters_
 
corrplot.mixed(cor(auto[, -8]), upper="circle")
  

##**Comment** : From the corrplot mpg01 has high correlation with cylinders,displacement,horsepower and weight 

#_Using Scatterplot to see any relationship between the variables_

 
cols <- character(nrow(auto))
cols[auto$mpg01 == 0]<-"purple2"
cols[auto$mpg01 == 1] <- "olivedrab2"

#Scatterplot
pairs(~mpg01+cylinders+displacement+horsepower+weight+acceleration+year,data = auto,col=cols,pch = 16,cex = 0.75)
  
#**Comment**: There is a postive relationship between displacement,horsepower and weight among each other and they have a negative relationship with acceleration

#_Boxplots_
 
par(mfrow=c(2,3))
boxplot(cylinders ~ mpg01, data = auto, main = "Cylinders vs mpg01",col="green")
boxplot(displacement ~ mpg01, data = auto, main = "Displacement vs mpg01",col = "red")
boxplot(horsepower ~ mpg01, data = auto, main = "Horsepower vs mpg01",col = "magenta")
boxplot(weight ~ mpg01, data = auto, main = "Weight vs mpg01",col = "paleturquoise1")
boxplot(acceleration ~ mpg01, data = auto, main = "Acceleration vs mpg01",col = "orange")
boxplot(year ~ mpg01, data = auto, main = "Year vs mpg01",col = "lightslateblue")


  


##**(c) Split the data into a training set and a test set**##

#_Splitting the data into half training and half testing_

 
set.seed(1)
index = sample(1:nrow(auto),nrow(auto)*0.5)
auto_train = auto[index,]
auto_test = auto[-index,]
mpg01_train = mpg01[index]
mpg01_test = mpg01[-index]

  


##**(d) Perform LDA on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?**##

#_Performing Linear Discriminant Analysis and getting the test error for the variables that are most associated with mpg01_
 
#LDA
lda.fit=lda(mpg01~cylinders+displacement+horsepower+weight+acceleration+year, data=auto_train)
lda.fit
plot(lda.fit)
lda.pred=predict(lda.fit,auto_test)$class
table(lda.pred,auto_test$mpg01)
mean(lda.pred!=auto_test$mpg01)
  
##**Comment**: We may conclude that, for Linear Discriminant Analysis, we have a test error rate of 12.7551%.

##**(e) Perform QDA on the training data in order to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?**##

#_Performing Quadratic Discriminant Analysis and getting the test error for the variables that are most associated with mpg01_
 
#QDA
qda.fit=qda(mpg01~cylinders+displacement+horsepower+weight+acceleration+year, data=auto_train)
qda.fit
qda.pred=predict(qda.fit,auto_test)$class
table(qda.pred,auto_test$mpg01)
mean(qda.pred!=auto_test$mpg01)
  
##**Comment**: We may conclude that, for Quadratic Discriminant Analysis, we have a test error rate of 9.693878%.

##**(f) Perform logistic regression on the training data to predict mpg01 using the variables that seemed most associated with mpg01 in (b). What is the test error of the model obtained?**##

#_Performing Logistic Regression and getting the test error for the variables that are most associated with mpg01_
 
glm.fit=glm(mpg01~cylinders+displacement+horsepower+weight+acceleration+year, data=auto_train,family = binomial)
glm.fit
par(mfrow = c(2,2))
plot(glm.fit)
auto.probs = predict(glm.fit, auto_test, type = "response",)
auto.pred = rep(0, length(auto.probs))
auto.pred[auto.probs > 0.5] = 1
table(auto.pred, mpg01_test)
mean(auto.pred != mpg01_test)
  
##**Comment**: We may conclude that, for Logistic Regression, we have a test error rate of 8.673469%.

##**(h) Perform KNN on the training data, with several values of K, in order to predict mpg01. Use only the variables that seemed most associated with mpg01 in (b). What test errors do you obtain? Which value of K seems to perform the best on this data set?**##

#_Performing KNN with K values from 1 to 10 and getting the test error for the variables that are most associated with mpg01_
 
auto_training = cbind(auto_train$cylinders+auto_train$displacement+auto_train$horsepower+auto_train$weight+auto_train$acceleration+auto_train$year)
auto_testing = cbind(auto_test$cylinders+auto_test$displacement+auto_test$horsepower+auto_test$weight+auto_test$acceleration+auto_test$year)
knn_pred = NULL
error_rate = NULL
for(i in 1:10){
set.seed(1)
knn_pred = knn(auto_training,auto_testing,mpg01_train,k=i)
error_rate[i] = mean(mpg01_test != knn_pred)
pred=knn(auto_training, auto_testing, mpg01_test, k=i)
print(table(pred,mpg01_test))
}
error_rate
plot(error_rate)
min(error_rate)
which.min(error_rate)

  

##**Comment**: The error rate is minimum for 8 variables with error rate as 11.22449%. After reviewing the results of each classification method, Logistic Regression has the least error of 8.673469% followed by Quadratic Discriminant Analysis. 


















