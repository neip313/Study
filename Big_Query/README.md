# Google_Big_Query_The_Definitive_Guide

![book](image/book_image.jpg)

## 책소개 
빅데이터, 데이터 엔지니어링, 머신러닝을 위한 대용량 데이터 분석과 처리의 모든 것

협업과 신속함을 갖춘 작업 공간을 구축하는 동시에 페타바이트 규모의 데이터셋을 처리해보자. 이 책은 기업 전체에서 추출한 데이터를 통합하고 대화형 데이터 분석과 대규모 데이터셋 기반의 머신러닝을 가능케 하는 쿼리 엔진을 제공하는 구글 빅쿼리에 대한 완벽 가이드다. 기업은 빅쿼리를 사용해 하나의 편리한 프레임워크로 데이터를 효율적으로 저장, 쿼리, 수집, 학습할 수 있다.

이 책의 저자 발리아파 락쉬마난과 조던 티가니는 공개 클라우드 상에서 자동으로 확장되는 서버리스 아키텍처에 기반한 최신 데이터 웨어하우징을 위한 모범 사례를 제시하고 있다. 이제 막 빅쿼리를 시작하면서 전반적인 기능을 훑어보고자 하는 독자는 물론 빅쿼리를 이용해 특정 업무를 해결하고자 하는 독자에게도 완벽한 가이드가 되어 줄 것이다.

[인터넷 교보문고 제공]

## 이 책에서 다루는 내용 
■ 빅쿼리의 고수준 아키텍처는 물론 내부 동작까지 상세한 가이드
■ 빅쿼리가 지원하는 데이터 타입, 함수, 연산자에 대한 설명
■ 쿼리 및 스키마 최적화를 통해 성능을 향상시키거나 비용을 절감하는 비법
■ GIS, 시간 여행, DDL/DML, 사용자 정의 함수, 표준 SQL 내의 스크립팅 등 고급 기술 학습
■ 빅쿼리 ML로 다양한 머신러닝 문제를 해결하는 방법
■ 데이터를 보호하고 작업을 모니터링하며 사용자를 인증하는 방법
■ 스크립팅, 예약, 구체화된 뷰, 컬럼 수준 보안, 동적 SQL, 머신러닝, 테이블 수준 접근 제어, 통합 쿼리 등 최신 기술 업데이트

## 환경 설정
-  『구글 빅쿼리 완벽 가이드』의 예제 코드를 실행하려면 다음과 같은 환경 설정이 필요합니다. 원서에는 담겨 있지 않은 내용이라 국내 독자를 위해 정리해뒀습니다. 
---


### bq 명령줄 도구(command-line) 설치
- Google Cloud SDK 설치시 사용 가능
- [Installing Google Cloud SDK](https://cloud.google.com/sdk/docs/install?hl=ko)로 이동
	- Mac OS : 비트에 맞는 파일을 설치한 후, 스크립트 실행

		```
		./google-cloud-sdk/install.sh
		```

	- 윈도우 : [Cloud SDK 설치 프로그램](https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe?hl=ko) 다운로드 또는 파워셀에서 아래 명령어 실행

		```
		(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
		& $env:Temp\GoogleCloudSDKInstaller.exe
		```

	- 공통
		- gcloud 초기화

		```
		gcloud init
		```



## 데이터셋 생성
- 데이터셋이 없는 경우, 테이블 생성이 불가능합니다. 데이터셋을 먼저 생성해주세요
- 7장 데이터셋 생성

	```
	bq --location=US mk ch07
	bq --location=US mk ch07eu
	```


- 8장 데이터셋 생성

	```
	bq --location=US mk ch08eu
	```

- 9장 데이터셋 생성

	```
	bq --location=US mk ch09eu
	```

- 10장 데이터셋 생성

	```
	bq --location=US mk ch10eu
	```


## 목차 
|1장|구글 빅쿼리|||
|-|-|-|-|
|||데이터 처리 아키텍처||
||||관계형 데이터베이스 관리 시스템|
||||맵리듀스 프레임워크|
||||빅쿼리: 서버리스, 분산 SQL 엔진|
|||빅쿼리로 작업하기||
||||여러 데이터셋에서 통찰력 도출하기|
||||ETL, EL, ELT|
||||강력한 분석|
||||관리의 단순함|
|||빅쿼리는 어떻게 만들어졌는가||
|||빅쿼리는 어떻게 구현할 수 있었을까||
||||컴퓨팅 및 스토리지 분리|
||||스토리지 및 네트워킹 인프라|
||||관리형 저장소|
||||구글 클라우드 플랫폼과 통합|
||||보안 및 규정 준수|
|||정리||
|2장|쿼리 필수 요소|||
|||간단한 쿼리||
||||SELECT로 행 검색하기|
||||AS로 컬럼 이름에 별칭 지정하기|
||||WHERE로 필터링하기|
||||SELECT, EXCEPT, REPLACE|
||||WITH를 사용한 서브 쿼리|
||||ORDER BY로 정렬하기|
|||집계||
||||GROUP BY로 집계하기|
||||COUNT로 레코드 수 세기|
||||HAVING으로 그룹화된 항목 필터링하기|
||||DISTINCT로 고윳값 찾기|
|||배열과 구조체 기초||
||||ARRAY_AGG로 배열 만들기|
||||구조체의 배열|
||||튜플|
||||배열 활용하기
||||배열 풀기|
|||테이블 조인||
||||조인의 작동 원리|
||||이너 조인|
||||크로스 조인|
||||아우터 조인|
|||저장 및 공유||
||||쿼리 기록 및 캐싱|
||||저장된 쿼리|
||||뷰와 공유 쿼리의 비교|
|||정리||
|3장|데이터 타입, 함수, 연산자|||
|||숫자형과 함수||
||||수학 함수|
||||표준 규격 부동 소수점 분할|
||||SAFE 함수|
||||비교|
||||NUMERIC을 사용한 정밀 소수 계산|
|||불(BOOL) 다루기||
||||논리 연산|
||||조건식|
||||COALESCE로 NULL 값을 깨끗하게 처리하기|
||||타입 변환과 타입 강제|
||||불리언 변환을 피하기 위해 COUNTIF 사용하기|
|||문자열 함수||
||||국제화|
||||출력 및 파싱|
||||문자열 조작 함수|
||||변환 함수|
||||정규 표현식|
||||문자열 함수 정리|
|||타임스탬프 다루기||
||||타임스탬프 값의 파싱과 형식화|
||||달력 정보 추출하기|
||||타임스탬프 연산하기|
||||Date, Time 그리고 DateTime|
|||GIS 함수 사용하기||
||정리||
|4장|빅쿼리로 데이터 로드하기|||
|||가장 기본적인 방법||
||||로컬에서 데이터 로드하기|
||||스키마 지정하기|
||||새 테이블에 복사하기|
||||데이터 관리(DDL과 DML)|
||||데이터를 효율적으로 로드하기|
|||통합 쿼리와 외부 데이터 원본||
||||통합 쿼리 사용하기|
||||통합 쿼리와 외부 데이터 원본의 사용 사례|
||||대화형 탐색과 구글 시트 데이터의 쿼리|
||||클라우드 빅테이블의 데이터에 대한 SQL 쿼리|
|||전송과 내보내기||
||||데이터 전송 서비스|
||||스택드라이버 로그 내보내기|
||||클라우드 데이터플로우로 빅쿼리 데이터 읽고 쓰기|
|||온프레미스 데이터의 이동||
||||데이터 마이그레이션 방법|
|||정리||
|5장|빅쿼리를 활용한 개발|||
|||프로그래밍 방식을 활용한 개발||
||||REST API 활용하기|
||||구글 클라우드 클라이언트 라이브러리|
|||데이터 과학 도구에서 빅쿼리 사용하기||
||||구글 클라우드 플랫폼의 노트북|
||||빅쿼리, 판다스, 그리고 주피터의 결합|
||||R에서 빅쿼리 다루기|
||||클라우드 데이터플로우|
||||JDBC/ODBC 드라이버|
||||빅쿼리 데이터를 G 스위트의 구글 슬라이드에 포함하기|
|||빅쿼리와 배시 스크립팅||
||||데이터셋과 테이블 생성|
||||쿼리의 실행|
||||빅쿼리 객체|
|||정리||
|6장|빅쿼리 아키텍처|||
|||아키텍처 살펴보기||
||||쿼리 요청의 수명|
||||빅쿼리 업그레이드|
|||쿼리 엔진(드레멜)||
||||드레멜 아키텍처|
||||쿼리 실행|
|||스토리지||
||||스토리지 데이터|
||||메타데이터|
|||정리||
|7장|성능 및 비용 최적화|||
|||성능 최적화의 기본 원칙||
||||성능의 핵심 요소|
||||비용 통제하기|
|||측정과 문제 해결||
||||REST API로 쿼리 속도 측정하기|
||||빅쿼리 워크로드 테스터로 쿼리 속도 측정하기|
||||스택드라이버를 사용해 워크로드 문제 해결하기|
||||쿼리 실행 계획 정보 읽기|
||||작업 세부 정보에서 쿼리 계획 정보 가져오기|
||||쿼리 계획 정보 시각화|
|||쿼리 속도 높이기||
||||I/O 최소화|
||||이전 쿼리 결과 캐싱하기|
||||효율적으로 조인하기|
||||워커의 과도한 작업 피하기|
||||근사 집계 함수 사용하기|
|||데이터 저장 및 접근 방법 최적화||
||||네트워크 오버헤드 최소화하기|
||||효율적인 저장 포맷 선택하기|
||||스캔 크기를 줄이기 위해 테이블 파티셔닝하기|
||||높은 카디널리티 키에 기반한 클러스터링 테이블|
||||시간에 구애받지 않는 사용 사례|
|||정리||
||||체크리스트|
|8장|고급 쿼리|||
|||재사용 가능한 쿼리||
||||파라미터화된 쿼리|
||||SQL 사용자 정의 함수|
||||쿼리 일부 재사용하기|
|||고급 SQL||
||||배열 다루기|
||||윈도우 함수|
||||테이블 메타데이터|
||||데이터 정의 언어와 데이터 조작 언어|
|||SQL 이상의 기능||
||||자바스크립트 사용자 정의 함수|
||||스크립팅|
|||고급 함수||
||||빅쿼리 지리 정보 시스템|
||||유용한 통계 함수들|
||||해시 알고리즘|
|||정리||
|9장|빅쿼리 머신러닝|||
|||머신러닝이란||
||||머신러닝 문제 공식화하기|
||||머신러닝 문제의 유형|
|||회귀 모델 생성하기||
||||레이블 선택하기|
||||피처를 찾기 위한 데이터셋 탐색|
||||학습 데이터셋 생성하기|
||||모델 학습 및 평가|
||||모델로 예측하기|
||||모델 가중치 검사하기|
||||더 복잡한 회귀 모델|
|||분류 모델 생성하기||
||||학습|
||||평가|
||||예측|
||||임계값 고르기|
|||빅쿼리 ML 커스텀하기||
||||데이터 분할 제어하기|
||||클래스 균형 맞추기|
||||정규화|
|||k 평균 클러스터링||
||||어떤 것을 클러스터링할까|
||||자전거 대여소 클러스터링하기|
||||클러스터링 수행하기|
||||클러스터 이해하기|
||||데이터 기반 의사 결정|
|||추천 시스템||
||||무비렌즈 데이터셋|
||||행렬 분해|
||||추천 만들기|
||||사용자와 영화 정보 통합하기|
|||GCP의 커스텀 머신러닝 모델||
||||하이퍼파라미터 튜닝|
||||AutoML|
||||텐서플로우 지원|
|||정리||
|10장|빅쿼리 관리 및 보안|||
|||인프라스트럭처 보안||
|||계정 및 접근 관리||
||||계정|
||||역할|
||||리소스|
|||빅쿼리 관리||
||||작업 관리|
||||사용자에게 권한 부여|
||||삭제된 레코드와 테이블의 복구|
||||지속적 통합/지속적 배포|
||||대시보드와 모니터링, 그리고 감사 로깅|
|||가용성과 재해 복구, 암호화||
||||존과 리전 그리고 멀티리전|
||||빅쿼리와 장애 처리|
||||내구성과 백업 그리고 재해 복구|
||||개인정보 보호와 암호화|
|||규제의 준수||
||||데이터 지역성|
||||데이터의 서비스에 대한 접근 제한|
||||개인과 관련된 모든 트랜잭션 제거하기|
||||데이터 유실 방지|
||||CMEK|
||||데이터 유출 보호|
|||정리||
|한국어판 특별 부록| 클라우드 컴포저와 빅쿼리로 ELT 파이프라인 만들기|||
|||ELT 파이프라인의 큰 그림||
|||클라우드 컴포저란||
|||클라우드 컴포저 생성 및 환경 설정||
|||클라우드 컴포저 웹 서버 UI||
|||DAG 만들기||
|||ELT 파이프라인 만들기||

