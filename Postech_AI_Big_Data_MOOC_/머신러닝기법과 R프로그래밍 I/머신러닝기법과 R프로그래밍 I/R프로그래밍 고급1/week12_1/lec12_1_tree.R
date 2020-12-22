# lec12_1_tree.R
# Decision tree
# use package "tree"

# 기계학습 중 하나로 의사결정 규칙을 나무 형태로 분류해나가는 분석기법 
# 분석 과정이 직관적이고 이해하기 쉬움 
# 연속형 / 범주형 변수를 모두 사용할 수 있음
# 분지규칙은 불순도를 최소화시킴 
# 분지규칙 : 범주들이 섞여있는 정도  

# STEP1 : TREE 형성 (growing tree)
# STEP2 : TREE 가지치기 (pruning tree )
# STEP3 : 최적 TREE로 분류(Classification) 


#decision tree packages download
install.packages("tree")
#load library
library(tree)

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
#dim(train)
#dim(test)
library(dplyr) 
help("tree")

# 의사결정나무 함수 : tree(종속변수~x1+x2+x3+x4, data= )
# step1 : growing tree
# treemod는 iris데이터의 범주를 분리해주는 분지결과를 저장
# plot(treemod) – 의사결정나무 분지를 그림으로 표현
# cex-폰트 사이즈 (1)
treemod<-tree(Species~., data=train)
treemod
# tree의 결과 (*는 터미널노드) : 마디 6에서는 더이상 분지할 필요 없음
plot(treemod)
text(treemod,cex=1.5)

table(train$Species)

# step2 : pruning using cross-validation
# 최적 tree 모형을 위한 가지치기(pruning) : cv.tree(tree모형결과, FUN= )
# 아래 결과에서 복잡도 계수(cost complexity parameter)의 값이 최소가 되는 노드수 선택 
cv.trees<-cv.tree(treemod, FUN=prune.misclass)
cv.trees
plot(cv.trees)
# k는 복잡도계수(complexity parameter)

# final tree model with the optimal node 
prune.trees<-prune.misclass(treemod, best=3)
plot(prune.trees)
text(prune.trees,pretty=0, cex=1.5)
#help(prune.misclass)

# step 3: classify test data 
treepred<-predict(prune.trees,test,type='class')
# classification accuracy
confusionMatrix(treepred,test$Species)

