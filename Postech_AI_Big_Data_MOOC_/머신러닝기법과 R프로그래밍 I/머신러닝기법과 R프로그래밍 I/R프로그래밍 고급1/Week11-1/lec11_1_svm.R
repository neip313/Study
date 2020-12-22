# lec11_1_svm.r
# Classification
# support vector machine (e1071)

# install package for support vector machine
install.packages("e1071")
library (e1071)
#help(svm)

# set working directory


# read data
iris<-read.csv("iris.csv")
attach(iris)


## classification 
# 1. use all data 
m1<- svm(Species ~., data = iris, kernel="linear")

summary(m1)
# svm에서 주어지는 옵션(default)
# kernal = radial basis function, gamma=1/(# of dimension)(1/4=0.25)

# classify all data using svm result (m1)
# first 4 variables as attribute variables
 
x<-iris[, -5] # iris데이터에서 타겟값인 5번쨰 열을 제외한 데이터, 즉 4개의 독립변수들만 있는 데이터
pred <- predict(m1, x) # svm모델 m1을 적용하여 예측된 범주값을 pred로 저장

# Check accuracy (compare predicted class(pred) and true class(y))
# y <- Species or y<-iris[,5]
y<-iris[,5]
table(pred, y)

# visualize classes by color
plot(m1, iris,  Petal.Width~Petal.Length, slice=list(Sepal.Width=3, Sepal.Length=4))


