# 스팀 리뷰 감성분석 서비스   
  
주제 : 게임  
수집 사이트 : 스팀  
수집 대상 : 게임 리뷰  
수집 조건 : 게임 당 리뷰 5천건 제한  
현재 수집 상황 : 인기게임, RPG, 인디, 어드벤처, 시뮬레이션, 스포츠 장르 각 최대 50만건 수집 완료   
중복 게임 제거(한 게임의 장르가 여러개라 중복값 존재) 및 한글 리뷰 필터링(한국 국적으로 영어, 이모티콘으로 이루어진 리뷰 존재) 진행 후 최종 리뷰 115만건  

## 실행방법   
# 1. 데이터, 모델 다운로드   
[크롤링 데이터 다운로드](https://github.com/PlantJelly/text_data_crawler/releases/download/v260701/merged_cleaned_reviews.csv)   
/data 폴더 속에 저장   
[감성분석 데이터 다운로드](https://github.com/PlantJelly/text_data_crawler/releases/download/v260701/lstm_sentiment.csv)   
/data 폴더 속에 저장   
[감성분석 모델 다운로드](https://github.com/PlantJelly/text_data_crawler/releases/download/v260701/sa_model_game.keras)   
/model 폴더 속에 저장   
[딥러닝 데이터 다운로드](https://github.com/PlantJelly/text_data_crawler/releases/download/v260701/sa_tokenizer_game.pkl)   
/model 폴더 속에 저장   

# 2. 실행   
루트폴더 속 **start.bat** 실행 시 가상 환경 생성 -> 필수 라이브러리 설치 -> 스트림릿 구동 순으로 실행
