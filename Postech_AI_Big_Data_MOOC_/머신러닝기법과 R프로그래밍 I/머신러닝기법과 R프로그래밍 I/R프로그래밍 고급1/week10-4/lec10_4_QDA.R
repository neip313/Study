# lec10_4_QDA.R
# Quadratic Discriminant Analysis

# MASS package for LDA
# install.packages("MASS")
library(MASS)

# install.packages("gmodels") #crosstable
library(gmodels)

# set working directory

# read csv file
iris<-read.csv("iris.csv")
attach(iris)

# training/ test data : n=150
set.seed(1000)
N=nrow(iris)
tr.idx=sample(1:N, size=N*2/3, replace=FALSE)

# attributes in training and test
iris.train<-iris[tr.idx,-5]
iris.test<-iris[-tr.idx,-5]
# target value in training and test
trainLabels<-iris[tr.idx,5]
testLabels<-iris[-tr.idx,5]

train<-iris[tr.idx,]
test<-iris[-tr.idx,]

# Box's M-test for Homogenity of Covariance Matrices
# 모집단 등분산 검정 : 분산-공분산 행렬이 범주별로 다른 경우, 이차판별분석(QDA)을 실시 
# 공분산 행령 -> Box's M-test
# 귀무가설 : 모집단의 분산 - 공분산 행렬이 동일 
# 대립가설 : 모집단의 분산 - 공분산 행렬이 동일 x  
# 등분산검정을 위한 패키지 설치
install.packages("biotools")
library(biotools)
boxM(iris[1:4], iris$Species)
# p-value~0 -> 귀무가설(등분산 가정)이 기각 -> QDA 실시! 

# Quadratic Discriminant Analysis (QDA)
# QDA 함수 : qda(종속변수~독립변수, data=학습 데이터 이름, prior=사전확률)
iris.qda <- qda(Species ~ ., data=train, prior=c(1/3,1/3,1/3))
iris.qda

# predict test data set n=50
testpredq <- predict(iris.qda, test)
testpredq

# accuracy of QDA
CrossTable(x=testLabels,y=testpredq$class, prop.chisq=FALSE)

# partimat() function for LDA & QDA
# Partition Plot : partimat()
install.packages("klaR")
library(klaR)
partimat(as.factor(iris$Species) ~ ., data=iris, method="lda")
partimat(as.factor(iris$Species) ~ ., data=iris, method="qda")





# stacked histogram
ldahist(data=testpred$x[,1], g=iris$Species, xlim=range(-10:10), ymax=0.7)
ldahist(data=testpred$x[,2], g=iris$Species, xlim=range(-10:10), ymax=0.7)

