# lec9_4_tr_test.R
# classification 
# training data and test data

# set working directory

# read csv file
iris<-read.csv(file="iris.csv")
head(iris)
str(iris)
attach(iris)

# training/ test data : n=150
set.seed(100) # seed 넘버는 내 맘대로 줘도 된다. 
N=nrow(iris)
tr.idx=sample(1:N, size=N*2/3, replace=FALSE) # 150개 데이터 중에서 2/3를 랜덤하게 샘플링한 데이터 추출 -> 3fold 가 됨 
tr.idx

# attributes in training and test
iris.train<-iris[tr.idx,-5] # 5번째 열의 종속변수를 제외한 100개의 데이터
iris.test<-iris[-tr.idx,-5] # 5번째 열의 종속변수를 제외한 50개의 데이터 

# target value in training and test
trainLabels<-iris[tr.idx,5] # train의 ?타켓변수의 값이 무엇이냐 -> tr.idx의 5번째 칼럼 
testLabels<-iris[-tr.idx,5] # test의 타켓변수 값이 무엇이냐

# to get frequency of class in test set
table(testLabels)  
