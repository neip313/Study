# lec14_1_ass.r 연관규칙 분석 I(Association Rule Analysis)
# Association Rule
# Market basket analysis

#• 연관규칙 (Association Rule)
#• 대용량 데이터베이스의 트랜잭션에서 빈번하게 발생하는 패턴을 발견
#• 거래간의 상호 관련성을 분석
#• 연관규칙 예시
#• 신발을 구매하는 고객의 10%는 양말을 동시에 구입한다.
#• 빵과 우유를 고매한 고객의 50%가 쥬스도 함께 구매한다.

#• 시장바구니(market basket) : 고객이 구매한 물품에 대한 정보 (구매 시기, 지불 방법, 매장정보 포함)
#• 트랜잭션(transaction) : 고객이 거래한 정보를 하나의 트랜잭션
#• 시장바구니 분석(market basket analysis) : 시장바구니 데이터로부터 연관규칙을 탐색 분석

# • 연관규칙을 평가하기 위해 지지도(support), 신뢰도(Confidence), 향상도(Lift)를 사용
# 지지도(Support) : A와 B를 동시에 포함하는 거래수/전체 거래수
# 신뢰도(Confidence): A와 B를 동시에 포함하는 거래 수/A를 포함하는 거래수
# 향상도(Lift) : A와 B를 동시에 포함하는 거래수/A를 포함하는 거래수 X B를 포함하는 거래수
#- 지지도가 어느 정도 수준에 도달해야만 한다. (A항목 지지도=A거래건수/전체거래수)
#- 신뢰도가 높을 경우에는 두 항목 A→B에서 항목 B의 확률이 커야지 연관규칙이 의미가 있다.
#- 향상도가 1보다 큰 값을 주어야 유용한 정보를 준다고 볼 수 있다.

#  향상도(lift) : A가 거래된 경우, 그 거래가 B를 포함하는 경우와 B가 임의로 거래되는 경우의 비율
#- 각 항목의 구매가 상호 관련이 없다면 P(B|A)와 P(B)와 같게 되어 향상도는 1이 됨
#- 1보다 크면 결과 예측에 대하여 우연적 기회(random chance)보다 우수함을 의미
#- 향상도의 값이 클수록 A의 거래 여부가 B의 거래 여부에 큰 영향을 미침

# set working directory
setwd("D:/tempstore/moocr/wk14")

# association rule analysis package
install.packages("arules")
library(arules)

# data import-> make transaction data
dvd1<-read.csv("dvdtrans.csv")
dvd1
dvd.list<-split(dvd1$Item,dvd1$ID) # Split을 통해 id별로 item들을 as함수를 통해 transaction 데이터로 변환
dvd.list
dvd.trans<-as(dvd.list,"transactions")
# arules package를 통해 transaction 데이터 변환과 연관 규칙 분석을 함
dvd.trans

inspect(dvd.trans)

# summary of dvd.trans
summary(dvd.trans)
# • 10트랜잭션 / 10 항목
# • 밀도가 0.3 라고 되어 있는데, 10*10 cell 중에서 30%의 cell에 거래가 발생해 숫자가 차 있다는 뜻
# • 거래항목 중 Gladiator=7번, Patriot=6번, Six Sense=6번 순으로 나왔음을 의미

# for running dvdtras data
dvd_rule<-apriori(dvd.trans,
                  parameter = list(support=0.2,confidence = 0.20,minlen = 2))
# support=0.2, confidence=0.2이상인 13개의연관규칙 생성됨
dvd_rule

# same code with short command
# dvd_rule<-apriori(dvd.trans, parameter = list(supp=0.2,conf= 0.20,minlen = 2))                             

summary(dvd_rule)
inspect(dvd_rule)

# Bar chart for support>0.2
itemFrequencyPlot(dvd.trans,support=0.2,main="item for support>=0.2", col="green")



