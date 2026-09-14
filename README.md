# Steam Review Analytics

Steam 게임 리뷰를 수집·전처리하고 감성 및 키워드를 분석해 보여주는 Streamlit 대시보드입니다.

Steam API와 SteamSpy를 이용해 데이터를 수집했으며, 중복 제거와 한글 리뷰 필터링을 거쳐 **약 115만 건의 리뷰**를 분석 대상으로 구성했습니다.

- 기간: 2026.05.13 ~ 2026.07.01
- 언어: Python
- Dashboard: Streamlit, Plotly
- NLP: KoNLPy(Okt), Kiwi
- Sentiment: TensorFlow/Keras LSTM

---

## 데이터 수집 및 전처리

SteamSpy에서 장르별 게임 목록을 가져오고 Steam Review API를 이용해 게임당 최대 5,000건의 한국어 리뷰를 수집합니다.

대량 수집 과정에서 커서가 반복되며 같은 리뷰가 다시 들어오는 문제가 있어 수집 기준을 최신순(`recent`)으로 변경하고, recommendation ID와 리뷰 본문을 기준으로 중복을 제거했습니다.

API 요청 제한에 대응하기 위해 429 응답 발생 시 대기 시간을 늘리는 가변 sleep/backoff를 적용했으며, 진행 상태를 파일에 저장해 수집이 중단되어도 이어서 실행할 수 있도록 구성했습니다.

전처리 단계에서는 BBCode와 불필요한 공백을 제거하고, 한글 비율이 일정 기준 이상인 리뷰만 남깁니다.

---

## 감성 및 키워드 분석

수집한 리뷰의 `voted_up` 값을 학습 라벨로 사용해 긍정·부정 데이터를 균형 샘플링하고, Okt 형태소 분석 결과를 기반으로 양방향 LSTM 감성 분류 모델을 학습했습니다.

전체 데이터에 대한 감성 예측 결과는 미리 계산해 저장하고, Streamlit에서는 이를 불러와 빠르게 결과를 표시합니다.

대시보드의 키워드 분석에는 Kiwi를 사용합니다. 형태소 분석 결과를 캐싱해 같은 게임을 다시 분석할 때 반복 계산을 줄였습니다.

---

## 대시보드 기능

- 게임 검색 및 리뷰 수 확인
- 긍정·부정 리뷰 수와 긍정 비율 표시
- 긍정/부정 주요 키워드 시각화
- 키워드 클릭 시 해당 단어가 포함된 실제 리뷰 확인
- 기본·사용자 지정 불용어 관리
- 분석 결과 캐싱

---

## 실행 방법

### 1. 데이터와 모델 다운로드

아래 파일을 GitHub Releases에서 내려받아 각 폴더에 저장합니다.

- [크롤링 데이터](https://github.com/PlantJelly/text_data_crawler/releases/download/v260701/merged_cleaned_reviews.csv) → `data/`
- [감성분석 결과](https://github.com/PlantJelly/text_data_crawler/releases/download/v260701/lstm_sentiment.csv) → `data/`
- [감성분석 모델](https://github.com/PlantJelly/text_data_crawler/releases/download/v260701/sa_model_game.keras) → `model/`
- [토크나이저](https://github.com/PlantJelly/text_data_crawler/releases/download/v260701/sa_tokenizer_game.pkl) → `model/`

### 2. 실행

Windows에서는 루트의 `start.bat`을 실행하면 가상환경 생성, 의존성 설치, Streamlit 실행을 순서대로 진행합니다.

직접 실행할 경우:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 시연 영상

https://github.com/user-attachments/assets/e45b959b-4544-4eaf-9e03-0dff4597f4ec
