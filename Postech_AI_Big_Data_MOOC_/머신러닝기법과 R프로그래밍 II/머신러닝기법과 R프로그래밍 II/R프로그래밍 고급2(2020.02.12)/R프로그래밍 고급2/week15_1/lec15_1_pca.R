# lec15_1_pca.r
# Multivariate analysis
# 주성분분석 Principle Component Analysis
#  다변량분석기법
#  ‘주성분’이라고 불리는 선형조합으로 표현하는 기법
#  여기서 주성분은 공분산(XTX)으로부터 eigenvector와 eigenvalue를 도출하여 계산됨

#• 주성분간의 수직관계

#• 1st 주성분 (PC1) : 독립변수들의 변동(분산)을 가장 많이 설명하는 성분
#• 2nd 주성분 (PC2) : PC1과 수직인 주성분
# (첫번째 주성분이 설명하지 못하는 변동에 대해 두번째로 설명하는 성분)

# set working directory
setwd("D:/tempstore/moocr/wk15")

#input data(iris)
iris<-read.csv(file="iris.csv")
attach(iris)
head(iris)

#Check correlation
cor(iris[1:4])

# 1.PCA(center=T->mean=0, scale.=T->variance=1)
ir.pca<-prcomp(iris[,1:4],center=T,scale.=T)
ir.pca
summary(ir.pca)

# ir.pca is the weight to get 1st-4th principal compoenents 

# 2.scree plot : to choose the number of components
plot(ir.pca,type="l")

# either way to draw scree plot
screeplot(ir.pca)


# 3. calculate component=x_data%*% PCA weight
PRC<-as.matrix(iris[,1:4])%*%ir.pca$rotation
head(PRC)

# 4. classification using principal components
# make data with components
iris.pc<-cbind(as.data.frame(PRC), Species)
head(iris.pc)

# 5. support vector machine
# install.packages("e1071")
library (e1071)

# classify all data using PC1-PC4 using support vector machine
m1<- svm(Species ~., data = iris.pc, kernel="linear")
# m2<- svm(Species ~PC1+PC2, data = iris.pc, kernel="linear")
summary(m1)

# predict class for all data 
x<-iris.pc[, -5]
pred <- predict(m1, x)
# check accuracy between true class and predicted class
y<-iris.pc[,5]
table(pred, y)


