# lec12_2_tree.R
# Decision tree
# use package rpart and party

# other package for tree
install.packages("rpart")
install.packages("party")
library(rpart)
library(party)

#package for confusion matrix
#install.packages("caret")
library(caret)

# set working directory
setwd("D:/tempstore/moocr/wk12")

# read csv file
iris<-read.csv("iris.csv")
attach(iris)

# training (n=100)/ test data(n=50) 
set.seed(1000)
N<-nrow(iris)
tr.idx<-sample(1:N, size=N*2/3, replace=FALSE)
# split train data and test data
train<-iris[tr.idx,]
test<-iris[-tr.idx,]

#decision tree : use rpart package
help("rpart")

# 의사결정나무 함수 : rpart (종속변수~x1+x2+x3+x4, data= )
cl1<-rpart(Species~., data=train)
plot(cl1)
text(cl1, cex=1.5)
# rpart 함수는 가지치기를 해서 나온 결과 -> 데이터에 따라 부가적인 가지치기가 필요할 수도 있음
# * tree패키지에서 pruning한 결과와 동일

# rpart패키지는 과적합의 우려가 있으므로 pruning을 해줘야 함(iris의 경우 필요없음)
# printcp에서 xerror(cross validation error)의 값이 최소가 되는 마디를 선택 -> 3 
#pruning (cross-validation)-rpart
printcp(cl1)
plotcp(cl1)
# xerror(cross validation error)의 값이 최소가 되는 마디를 자동으로 선택
pcl1<-prune(cl1, cp=cl1$cptable[which.min(cl1$cptable[,"xerror"]),"CP"])
plot(pcl1)
text(pcl1)
# rpart결과에서 복잡도계수에 기반한 최적 가지치기

#measure accuracy(rpart)
pred2<- predict(cl1,test, type='class')
confusionMatrix(pred2,test$Species)

#decision tree(party)-unbiased recursive partioning based on permutation test
partymod<-ctree(Species~.,data=train)
plot(partymod)

#measuring accuracy(party)
partypred<-predict(partymod,test)
confusionMatrix(partypred,test$Species)