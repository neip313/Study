# lec3_4.r
# Reading other format data : EXCEL, SPSS, SAS, ODBC

# set working directory
setwd("D:/tempstore/moocr")

# 1. several sheets in Excel file
install.packages("readxl")
library(readxl)

mtcars1 <- read_excel("D:/tempstore/moocr/mtcarsb.xlsx", 
                      sheet = "mtcars")
mtcars1 <- read_excel("D:/tempstore/moocr/mtcarsb.xlsx", 
                      sheet = 1)
head(mtcars1)

brain1<-read_excel("D:/tempstore/moocr/mtcarsb.xlsx", 
                 sheet = "brain")
head(brain1)

brainm<-read_excel("D:/tempstore/moocr/mtcarsb.xlsx", 
                   sheet = 2)
head(brainm)

# 2. ODBC data import- STATA, Systat, Weka, dBase ..
install.packages("foreign")
library(foreign)

# 3. Reading SAS data file
install.packages("sas7bdat")
library(sas7bdat)

b1<-read.sas7bdat("brain.sas7bdat")
head(b1)
str(b1)

# 4. Reading from SQL database
# by Anders Stockmarr, Kasper Kristensen

# reading data from SQL databse

install.packages("RODBC")
library(RODBC)

connStr <- paste(
  "Server=msedxeus.database.windows.net",
  "Database=DAT209x01",
  "uid=RLogin",
  "pwd=P@ssw0rd",
  "Driver={SQL Server}",
  sep=";"
)

conn <- odbcDriverConnect(connStr)

#A first query

tab <- sqlTables(conn)
head(tab)

#Getting a table

mf <- sqlFetch(conn,"bi.manufacturer")
mf

close(conn)

##################################################################
##################################################################

set1 <- select(car, mpg, hp)
head(set1)

set2 <- select(car, -starts_with("mpg")) # mpg 를 제외한 나머지 변수 
head(set2)

# 조건식에 맞는 데이터 추출 : filter(데이터, 변수조건, ...)
# car 데이터에서 mpg가 30보다 큰 행 추출 
set3 <- filter(car, mpg > 30)
head(set3)

# 4. 변수생성 : mutate(새로운 변수이름 : 기존 변수 활용)
#  %>% (파이프 연산자) 연산자 사용하여 연결 
set4 <- car %>% 
  filter(!is.na(mpg)) %>% 
  mutate(mpg_km = mpg*1.609) # mile을 km로 변경(1.609를 곱해준다)

head(set4)

# 5. 데이터 요약통계치 
# 데이터 요약통계치(평균 구하기) : summarize(mean(변수이름))
# mpg, hp, wt의 평균값 구하기 
car %>% 
  summarize(mean(mpg), mean(hp), mean(wt))

# 몇 개 변수들의 평균값 한번에 구하기 
select(car, 1:6) %>% 
  colMeans()

# 벡터화 요약치 : summarize_all(FUN)
# 열 추출하여 기술통계치 구하고 요약치 보여줌 
a1 <- select(car, 1:6) %>%  summarize_all(mean)
a2 <- select(car, 1:6) %>%  summarize_all(sd)
a3 <- select(car, 1:6) %>%  summarize_all(min)
a4 <- select(car, 1:6) %>%  summarize_all(max)
table1 <- data.frame(rbind(a1, a2, a3, a4))
rownames(table1) <- c("mean", "sd", "min", "max")
table1

# 그룹별 통계량 얻기 : group_by(변수), summarize(_=FUN())
# 그룹별 요약통계량 구하기 
car %>% 
  group_by(cyl) %>%  # car 데이터의 cyl열을 그룹으로 묶음 
  summarise(mean_mpg = mean(mpg, na.rm = TRUE)) # cyl그룹의 mpg 평균을 구함 
# 통계 분석 시 결측값을 제외함 

237710+1472801+59925+162838+147895
