import numpy as np

_MAX_LEN = 80   # train_game_model.py MAX_LEN 과 일치해야 함


class SentimentAnalyzer:
    """LSTM 기반 감성 분석 래퍼.

    모델 파일이 없으면 is_available=False로 설정되어
    voted_up 기반 분류만 사용하도록 fallback 처리된다.
    """

    def __init__(self, model_file: str = None, tokenizer_file: str = None):
        self.model = None
        self.tokenizer = None
        self.is_available = False
        self._morphs = None

        if not model_file or not tokenizer_file:
            return

        try:
            from tensorflow.keras.models import load_model
            import joblib
            from konlpy.tag import Okt

            self.model = load_model(model_file)
            self.tokenizer = joblib.load(tokenizer_file)
            self._morphs = Okt().morphs
            self.is_available = True
            print("LSTM 모델 로딩 완료")
        except FileNotFoundError as e:
            print(f"모델 파일을 찾을 수 없습니다: {e}")
        except Exception as e:
            print(f"LSTM 모델 로딩 실패 (voted_up 기반 분류 사용): {e}")

    def analyze_sentiment(self, text: str) -> tuple:
        """단일 텍스트의 감성을 분석한다.

        반환값: (label: str, probability: float)
            label: '긍정' or '부정', is_available=False이면 (None, 0.0)

        모델 출력 형태에 따라 자동으로 판별 방식을 선택한다.
        - shape (1,) : sigmoid 이진 분류 → 0.5 기준 threshold
        - shape (2,) : softmax 다중 분류 → argmax
        """
        if not self.is_available:
            return None, 0.0

        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            tokens = self._morphs(str(text))
            encoded = self.tokenizer.texts_to_sequences([tokens])
            X = pad_sequences(encoded, maxlen=_MAX_LEN, truncating='post', padding='post')
            preds = self.model.predict(X, verbose=0)

            if preds.shape[-1] == 1:
                # sigmoid 이진 출력: 1 = 긍정, 0 = 부정
                prob_pos = float(preds[0][0])
                if prob_pos > 0.5:
                    return '긍정', prob_pos
                else:
                    return '부정', 1.0 - prob_pos
            else:
                # softmax 2-class 출력: index 0 = 부정, index 1 = 긍정
                labels = ['부정', '긍정']
                idx = int(np.argmax(preds[0]))
                return labels[idx], float(preds[0][idx])

        except Exception as e:
            print(f"감성 분석 오류: {e}")
            return None, 0.0

    def analyze_batch(self, reviews: list, batch_size: int = 256) -> list:
        """리뷰 리스트를 배치로 처리해 감성 라벨 리스트를 반환한다.

        전체 형태소 분석 → 인코딩 → model.predict 1회 호출 순서로
        1건씩 루프하는 방식보다 10-20배 빠르다.
        """
        if not self.is_available:
            return [None] * len(reviews)

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # 1. 전체 형태소 분석
        try:
            from tqdm import tqdm
            it = tqdm(reviews, desc="형태소 분석", unit="건")
        except ImportError:
            it = reviews

        tokenized = []
        for review in it:
            try:
                tokenized.append(self._morphs(str(review)))
            except Exception:
                tokenized.append([])

        # 2. 전체 시퀀스 변환 + 패딩 (1회)
        sequences = self.tokenizer.texts_to_sequences(tokenized)
        X = pad_sequences(sequences, maxlen=_MAX_LEN, truncating='post', padding='post')

        # 3. 배치 예측 (model.predict 1회 호출)
        preds = self.model.predict(X, batch_size=batch_size, verbose=1)

        # 4. 라벨 변환 (sigmoid vs softmax 자동 판별)
        if preds.shape[-1] == 1:
            return ['긍정' if p[0] > 0.5 else '부정' for p in preds]
        else:
            return [['부정', '긍정'][int(np.argmax(p))] for p in preds]
