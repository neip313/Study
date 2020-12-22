# lec13_2_clus.R
# Clustering
# Hierarchical Clustering
# Linkage method, Dendrogram

# 1. 계층적 군집분석 : 사전에 군집 수 k를 정하지 않고 단계적으로 군집을 형성한다. 
# - 유사한 객체들을 군집으로 묶고, 그 군집을 기반으로 그와 유사한 군집을 새로운 군집으로 묶어가면서 군집을 계층적으로 구성함
# 계층적 군집 (단일연결법, 완전연결법, 평균연결법, 중심연결법)

# 2. 단일연결법 : 군집 i와 군집 j의 유사성 척도로 두 군집의 모든 객체 쌍의 거리 중 가장 가까운 거리를 사용 
# - 객체 쌍의 가장 짧은 거리가 작을수록 두 군집이 더 유사하다고 평가 
# 완전연결법(complete linkage method) : 두 군집의 모든 객체 상의 거리 중 가장 먼 거리를 사용 
# 평균연결법(average linkage method) : 두 군집의 모든 객체 쌍의 평균 거리를 사용 
# 중심연결법(centroid linkage method) : 두 군집의 중심 좌표 

# needs "lattice", "DAAG" package for loading dataset
# install.packages("lattice")
install.packages("DAAG")
library(lattice)
library(DAAG)

# load data in DAAG package
# the wages of Lancashire cotton factory workers in 1833
data("wages1833")
#help("wages1833")
head(wages1833,n=10)

# remove observations with the missing values
dat1<-wages1833
dat1<-na.omit(dat1) # 결측치가 있는 데이터의 행을 삭제(전처리)
str(dat1)
#data<-na.omit("wages1833")

# calculate distance between each nodes
dist_data<-dist(dat1) # 유클리디안 거리 사용 

# prepare hierarchical cluster
# complete linkage method
# 계층적 군집분석 : hclust(거리계산결과, method=" ")
hc_a <- hclust(dist_data, method = "complete") # 완전연결법 적용결과(거리 계산은 유클리디안 사용)
# single(단일), complete(완전), average(평균), centroid(중심) 
plot(hc_a, hang = -1, cex=0.7, main = "complete")

# average linkage method
# check how different from complete method
hc_c <- hclust(dist_data, method = "average") # 평균연결법 사용 
plot(hc_c, hang = -1, cex=0.7, main = "average")

# Ward's method
hc_c <- hclust(dist_data, method = "ward.D2") # 최근 기법으로 군집내의 제곱합을 이용 -> 주로 많이 사용 
plot(hc_c, hang = -1, cex=0.7, main = "Ward's method")

