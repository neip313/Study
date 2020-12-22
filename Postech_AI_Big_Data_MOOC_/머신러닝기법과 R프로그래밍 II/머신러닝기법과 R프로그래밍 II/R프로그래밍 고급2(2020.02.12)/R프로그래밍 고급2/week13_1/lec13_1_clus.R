# lec13_1_clus.R
# Clustering
# Distance measure

# similarity measures - distance
# 군집분석(cluster analysis)이란, 유사한 속성을 가진 객체들을 군집으로 나누는(묶어주는) 데이터마이닝 기법

# 군집분석의 방법은 
# (1) 계층적 방법(Hierarchical Clustering) : 사전에 군집 수 k를 정하지 않고 단계적으로 군집 트리를 제공 
# (2) 비계층적 방법(Non-hierarchical Clustering) : 사전에 군집 수 k를 정한 후 각 객체를 k개 중 하나의 군집에 배정 
# 으로 구분 

# 1. 유사성 척도 
# : 객체 간의 유사성 정도를 정량적으로 나타내기 위해서 척도가 필요
# 거리 (distance) 척도 : 거리가 가까울수록 유사성이 크다. 거리가 멀수록 유사성이 적어짐
# 상관계수척도 : 객체간 상관계수가 클수록 두 객체의 유사성이 커짐
m1 <- matrix(
  c(150, 50, 130, 55, 80, 80, 100, 85, 95, 91),
  nrow = 5,
  ncol = 2,
  byrow = TRUE)
# m1 is a matrix
# 데이터 생성 (m1, 5x2 행렬)
m1
is.data.frame(m1)
# m1 is defined as dataframe
m1<-as.data.frame(m1)
# m1을 data frame으로 저장 

# 1. Euclidean distance
# 디폴트값은 유클리디안 거리 
D1 <- dist(m1) 
D1

help("dist")

# 2. Minkowski distance
D2<- dist(m1, method="minkowski", p=3) 
D2

# 상관계수를 척도로 사용 
# 또 다른 유사성 척도로 객체 간의 상관계수를 사용 
# - 상관계수가 클수록 두 객체의 유사성이 크다고 추정 

# 상관계수측정(cor)
# 3. correlation coefficient
m2 <- matrix(
  c(20, 6, 14, 30, 7, 15, 46, 4, 2),
  nrow = 3,
  ncol = 3,
  byrow = TRUE) # 데이터 생성 (3x3 matrix)
m2

# 상관계수 측정 
# correlation between Obs1~Obs2
cor(m2[1,],m2[2,]) 
# correlation between Obs1~Obs3
cor(m2[1,],m2[3,])


