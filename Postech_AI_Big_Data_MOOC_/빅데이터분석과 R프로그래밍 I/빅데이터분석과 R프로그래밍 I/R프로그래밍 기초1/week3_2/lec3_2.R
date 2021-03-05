# lec3_2.r
# Data handling
# subset and export

# set working directory
# change working directory 
setwd("D:/tempstore/moocr")

brain<-read.csv("brain2210.csv")
head(brain)

attach(brain)

# subset with female
# brainf<-subset(brain, sex=='f') after attach(brain)
brainf<-subset(brain, sex=='f') 
mean(brainf$wt)
sd(brainf$wt)

# subset with male
brainm<-subset(brain, sex=='m') 
mean(brainm$wt)
sd(brainm$wt)

# subset with wt<1300
brain1300<-subset(brain,brain$wt<1300)

# same subset of brain1300
# brain1300<-subset(brain,!brain$wt>=1300)
summary(brain1300)

# 'aggregate'for statistics by group
aggregate(wt~sex, data=brain, FUN=mean) # 평균
aggregate(wt~sex, data=brain, FUN=sd) # 표준편차 

# histogram for female and male
# 2*2 multiple plot
par(mfrow=c(2,2))
brainf<-subset(brain,brain$sex=='f')# 여자아이들의 뇌 무게 히스토그램 
hist(brainf$wt, breaks = 12,col = "green",cex=0.7, main="Histogram (Female)" ,xlab="brain weight")

# subset with male
brainm<-subset(brain,brain$sex=='m') 
hist(brainm$wt, breaks = 12,col = "orange", main="Histogram with (Male)" ,  xlab="brain weight")

# histogram with same scale 
# X LIMIT을 정해주면 비교하기 쉽다
hist(brainf$wt, breaks = 12,col = "green",cex=0.7, main="Histogram with Normal Curve (Female)" , xlim=c(900,1700),ylim=c(0,25), xlab="brain weight")
hist(brainm$wt, breaks = 12,col = "orange", main="Histogram with Normal Curve (Male)" , xlim=c(900,1700), ylim=c(0,25),xlab="brain weight")

# plot margin
# par(mar=c(2,2,2,2))

# export csv file - write out to csv file 
write.table(brainf,file="brainf.csv", row.names = FALSE, sep=",", na=" ")

write.csv(brainf,file="brainf.csv", row.names = FALSE)

# export txt file 
write.table(brainm, file="brainm.txt", row.names = FALSE,  na=" ")

# QUIZ 1 
# 1.. brain 데이터에서 뇌의 무게가 <1000인 데이터를 생성하고, 남/녀 각각 몇 명이 있는지 답하시오 (lec3_2.r을 참고하여 brain1000이라는 subset을 생성하고, table함수를 이용하여 빈도를 구하라)

brainw<-subset(brain, wt<1000) 
table(brainw$sex)
