"""
Steam 게임 리뷰 기반 감성 분석 LSTM 모델 학습 스크립트

- 데이터: merged_cleaned_reviews.csv (115만 건)
- 라벨: voted_up (True=긍정, False=부정) — 학습용 라벨로만 사용
- 출력: model/sa_model_game.keras, model/sa_tokenizer_game.pkl

사용법:
    python train_game_model.py
"""

import os, sys, time
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# TF 경고 최소화
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATA_PATH          = os.path.join(BASE_DIR, 'data', 'merged_cleaned_reviews.csv')
MODEL_SAVE_PATH    = os.path.join(BASE_DIR, 'model', 'sa_model_game.keras')
TOKENIZER_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'sa_tokenizer_game.pkl')

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────
SAMPLE_PER_CLASS = 50000   # 긍/부 각각 5만 건 (총 10만 건) — 균형 샘플링
MAX_VOCAB        = 30000   # 어휘 크기
MAX_LEN          = 80      # 최대 시퀀스 길이 (게임 리뷰는 짧은 편)
EMBEDDING_DIM    = 128
LSTM_UNITS       = 64
BATCH_SIZE       = 512
EPOCHS           = 5
RANDOM_SEED      = 42


def tokenize_reviews(reviews: list, morphs_fn) -> list:
    """Okt.morphs로 리뷰 목록을 형태소 분석해 토큰 리스트 목록을 반환한다."""
    result = []
    for review in tqdm(reviews, desc="  형태소 분석", unit="건"):
        try:
            result.append(morphs_fn(str(review)))
        except Exception:
            result.append([])
    return result


def main():
    t0 = time.time()
    print("=" * 60)
    print("Steam 게임 리뷰 감성 분석 LSTM 모델 학습")
    print("=" * 60)

    # ── 1. 데이터 로딩 ───────────────────────────────────────────
    print(f"\n[1] 데이터 로딩...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    df.dropna(subset=['review'], inplace=True)
    if df['voted_up'].dtype == object:
        df['voted_up'] = df['voted_up'].map({'True': True, 'False': False})

    pos_df = df[df['voted_up'] == True]
    neg_df = df[df['voted_up'] == False]
    print(f"  전체: {len(df):,}건")
    print(f"  긍정: {len(pos_df):,}건 ({len(pos_df)/len(df)*100:.1f}%)")
    print(f"  부정: {len(neg_df):,}건 ({len(neg_df)/len(df)*100:.1f}%)")

    # ── 2. 균형 샘플링 ─────────────────────────────────────────
    print(f"\n[2] 균형 샘플링 (긍/부 각 {SAMPLE_PER_CLASS:,}건)...")
    n = min(SAMPLE_PER_CLASS, len(neg_df))
    pos_sample = pos_df.sample(n=n, random_state=RANDOM_SEED)
    neg_sample = neg_df.sample(n=n, random_state=RANDOM_SEED)
    samples = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=RANDOM_SEED)

    reviews = samples['review'].tolist()
    labels  = samples['voted_up'].astype(int).tolist()
    print(f"  학습 총 샘플: {len(samples):,}건")

    # ── 3. 형태소 분석 ────────────────────────────────────────
    print(f"\n[3] Okt 형태소 분석 ({len(reviews):,}건)...")
    from konlpy.tag import Okt
    okt = Okt()
    tokenized = tokenize_reviews(reviews, okt.morphs)

    # ── 4. Keras Tokenizer 구축 ───────────────────────────────
    print(f"\n[4] 어휘 구축 (최대 {MAX_VOCAB:,}개)...")
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(tokenized)
    actual_vocab = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    print(f"  실제 어휘 크기: {actual_vocab:,}")

    # ── 5. 시퀀스 변환 + 패딩 ─────────────────────────────────
    print(f"\n[5] 시퀀스 변환 및 패딩 (maxlen={MAX_LEN})...")
    sequences = tokenizer.texts_to_sequences(tokenized)
    X = pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
    y = np.array(labels, dtype=np.float32)

    # ── 6. 학습/검증 분할 ─────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  학습: {len(X_train):,} | 검증: {len(X_val):,}")

    # ── 7. 모델 정의 ──────────────────────────────────────────
    print(f"\n[6] LSTM 모델 정의...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential([
        Embedding(input_dim=actual_vocab, output_dim=EMBEDDING_DIM),
        Bidirectional(LSTM(LSTM_UNITS, dropout=0.2)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # ── 8. 학습 ───────────────────────────────────────────────
    print(f"\n[7] 학습 시작 (epochs={EPOCHS}, batch={BATCH_SIZE})...")
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=2,
            restore_best_weights=True, verbose=1,
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 9. 최종 평가 ───────────────────────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  최종 검증 정확도: {val_acc*100:.2f}%")
    print(f"  최종 검증 손실:   {val_loss:.4f}")

    # ── 10. 저장 ──────────────────────────────────────────────
    print(f"\n[8] 모델 저장...")
    os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    joblib.dump(tokenizer, TOKENIZER_SAVE_PATH)
    print(f"  모델:       {MODEL_SAVE_PATH}")
    print(f"  토크나이저: {TOKENIZER_SAVE_PATH}")

    elapsed = time.time() - t0
    print(f"\n총 소요 시간: {elapsed/60:.1f}분")
    print("=" * 60)
    print("학습 완료! app.py의 MODEL_PATH를 sa_model_game.keras로 변경하세요.")
    print("=" * 60)


if __name__ == '__main__':
    main()
