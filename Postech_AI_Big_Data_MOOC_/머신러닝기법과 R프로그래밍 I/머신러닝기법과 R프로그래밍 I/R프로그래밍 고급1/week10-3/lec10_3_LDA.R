# lec10_3_LDA.R
# Linear Discriminant Anlaysis

# install the MASS package for LDA
# 선형판별분석(LDA : Linear discriminant analysis) 패키지
install.packages("MASS")
library(MASS)

# install.packages("gmodels") #crosstable
library(gmodels)

# set working directory

# read csv file
iris<-read.csv("iris.csv")
attach(iris)
str(iris)
# training/ test data : n=150
set.seed(1000)
N=nrow(iris)
tr.idx=sample(1:N, size=N*2/3, replace=FALSE)
# 데이터분할(학습데이터 2/3, 검증데이터 1/3)

# attributes in training and test
iris.train<-iris[tr.idx,-5] #독립변수 4개를 포함한 100개의 데이터 
iris.test<-iris[-tr.idx,-5]#독립변수 4개를 포함한 50개의 데이터 

# target value in training and test
trainLabels<-iris[tr.idx,5] #학습데이터의 타겟변수 
testLabels<-iris[-tr.idx,5] #검증데이터의 타겟변수 

train<-iris[tr.idx,]
test<-iris[-tr.idx,]

# Linear Discriminant Analysis (LDA) with training data n=100
# LDA 함수 : lda(종속변수 ~ 독립변수, data = 학습데이터 이름, prior = 사전 확률)
iris.lda <- lda(Species ~ ., data=train, prior=c(1/3,1/3,1/3))
iris.lda

# predict test data set n=50
# 검증 데이터에 LDA 결과를 적용하여 범주 추정
# 어떻게 예측하는지 확인
testpred <- predict(iris.lda, test)
testpred

testpred1 <- round(testpred$posterior,2)
testpred1
# accuracy of LDA
# 정확도 산정 : 오분류율(검증데이터) 
CrossTable(x=testLabels,y=testpred$class, prop.chisq=FALSE)

# export csv file - write out to csv file 
write.table(testpred$posterior,file="posterior_iris.csv", row.names = TRUE, sep=",", na=" ")
write.table(test, ,file="test_iris.csv", row.names = TRUE, sep=",", na=" ")


# stacked histogram
# ldahist(data=testpred$x[,1], g=iris$Species, xlim=range(-10:10), ymax=0.7)
# ldahist(data=testpred$x[,2], g=iris$Species, xlim=range(-10:10), ymax=0.7)