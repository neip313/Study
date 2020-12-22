#lec13_3_clus.R
# Clustering
# Non-hierarchical Clustering

# 1. 비계층적 군집분석
# • 사전에 군집 수 k를 정한 후 각 객체를 𝐤개 중 하나의 군집에 배정
# ![](image/13-3-1.png)

# 2. k-means 군집분석
# k-means 군집분석은 비계층적 군집분석 중 가장 널리 사용
# - k개 군집의 중심좌표를 고려하여 각 객체를 가장 가까운 군집에 배정하는 것을 반복
# [단계 0] (초기 객체 선정) : k개 객체 좌표를 초기 군집 중심좌표로 선정.
# [단계 1] (객체 군집 배정) : 각 객체와 k개 중심좌표와의 거리 산출 후, 가장 가까운 군집에 객체 배정.
# [단계 2] (군집 중심좌표 산출) : 새로운 군집의 중심좌표 산출.
# [단계 3] (수렴 조건 점검) : 새로 산출된 중심 좌표값과 이전 좌표값을 비교.수렴 조건 내에 들면 종료, 그렇지 않으면 단계 1 반복.

# install package & set library
# install.packages("DAAG")
#library(DAAG)

# set working directory

# 데이터 불러오기 및 군집수 k 결정
# read csv file
wages1833<-read.csv(file="wages1833.csv")
head(wages1833)

# preprocessing
dat1<-wages1833
dat1<-na.omit(dat1)
head(dat1, n=5)

# to choose the optimal k
install.packages("factoextra")
library(factoextra)
fviz_nbclust(dat1, kmeans, method = "wss")
#  최적 군집수에 대한 시각화
#  최적값은 “silhouette”, “gap_stat”, “wss(그룹내합계제곱)” 으로 산출
#  그래프가 완만해지는 지점을 k의 값으로 추정

fviz_nbclust(dat1, kmeans, method = "gap_stat")

# compute kmeans
set.seed(123,sample.kind="Rounding")
km <- kmeans(dat1, 3, nstart = 25) # k-means (k=3) # random set의 수 (nstart)
km

km <- kmeans(dat1, 3, nstart=10)
km

km <- kmeans(dat1, 3)
km

# visualize
# Kmeans 결과 시각화
# Convex 모양으로 구역 표시
# Repel을 통해 관측치 표기
fviz_cluster(km, data = dat1, 
             ellipse.type="convex", 
             repel = TRUE)

# 5. K-medoids 군집분석
# • K-medoids 군집분석은 각 군집의 대표 객체(medoid)를 고려
# • 군집의 대표 객체란, 군집 내 다른 객체들과의 거리가 최소가 되는 객체
# • 즉, K-medoids 군집분석은 객체들을 K개의 군집으로 구분하는데,
# • 객체와 속하는 군집의 대표 객체와의 거리 총합을 최소로 하는 방법

# • PAM 알고리즘: 모든 객체에 대하여 대표 객체가 변했을 때 발생하는 거리 총합의 변화를 계산.
# 데이터 수가 많아질수록 연산량이 크게 증가함.
# • CLARA 알고리즘: 적절한 수의 객체를 샘플링 한 후, PAM 알고리즘을 적용하여 대표 객체 선정.
# 샘플링을 여러 번 한 후 가장 좋은 결과를 택함.
# 편향된 샘플링은 잘못된 결과값을 도출할 수 있음.


# 6. PAM (Partitioning Around Medoids) 알고리즘
# compute PAM
library("cluster")
pam_out <- pam(dat1, 3)
pam_out

# freq of each cluster
table(pam_out$clustering)

# visualize
fviz_cluster(pam_out, data = dat1,
             ellipse.type="convex", 
             repel = TRUE)


