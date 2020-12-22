# lec11_2_svm.r
# Classification 
# support vector machine using kernel
# 커널이란?
# - x의 기저함수(basis function)
# - x에 대한 새로운 특징을 추출하는 변환함수


# install package for support vector machine
# install.packages("e1071")
library (e1071)
# help(svm)

# install package for confusionMatrix
install.packages("caret")
library(caret)

# set working directory
setwd("D:/tempstore/moocr/wk11")

# read data
iris<-read.csv("iris.csv", stringsAsFactors = TRUE)
#str(iris)
attach(iris)

iris$Species <- as.factor(iris$Species)

str(iris)
iris_df <- iris
str(iris_df)

# training (100) & test set (50)
set.seed(1000)
N=nrow(iris)
tr.idx=sample(1:N, size=N*2/3, replace=FALSE)
# target variable
y=iris[,5]
# split train data and test data
train=iris[tr.idx,]
test=iris[-tr.idx,]

#svm using kernel
help("svm")
m1<-svm(Species~., data = test)
summary(m1)
m2<-svm(Species~., data = test,kernel="polynomial")
summary(m2)
m3<-svm(Species~., data = test,kernel="sigmoid")
summary(m3)
m4<-svm(Species~., data = test,kernel="linear")
summary(m4)

#measure accuracy
pred11<-predict(m1,test) # radial basis
confusionMatrix(pred11, test$Species)
#table(pred11, y[-tr.idx])

pred12<-predict(m2,test) # polynomial
confusionMatrix(pred12, test$Species)
#table(pred12, y[-tr.idx])

pred13<-predict(m3,test) # simoid
confusionMatrix(pred13, test$Species)
#table(pred13, y[-tr.idx])

pred14<-predict(m4,test) # linear
confusionMatrix(pred14, test$Species)
#table(pred14, y[-tr.idx])

CrossTable(test$Species, pred14)
