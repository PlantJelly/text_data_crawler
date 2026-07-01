"""
Steam 게임 리뷰 전체에 LSTM 감성 분석 적용 후 결과 저장 스크립트

- 입력  : data/merged_cleaned_reviews.csv  (115만 건)
- 출력  : data/lstm_sentiment.csv  (컬럼: idx, lstm_pos)
- 특징  : 중단 후 재실행 시 이어서 진행 (resume)
- 주의  : Okt 형태소 분석 병목으로 완료까지 수 시간 소요 (PC 성능에 따라 다름)
          완료 후 앱을 재시작해야 사전 계산 예측이 반영됩니다.

사용법:
    python precompute_sentiment.py
"""

import os
import sys
import time

import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(BASE_DIR, 'data', 'merged_cleaned_reviews.csv')
MODEL_PATH      = os.path.join(BASE_DIR, 'model', 'sa_model_game.keras')
TOKENIZER_PATH  = os.path.join(BASE_DIR, 'model', 'sa_tokenizer_game.pkl')
OUTPUT_PATH     = os.path.join(BASE_DIR, 'data', 'lstm_sentiment.csv')

CHUNK_SIZE = 5000   # 한 번에 처리할 리뷰 수 (메모리·재시작 단위)


def load_data():
    """load_game_reviews 와 동일한 방식으로 데이터를 로드한다 (인덱스 일치를 보장)."""
    df = None
    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(DATA_PATH, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"파일을 읽을 수 없습니다: {DATA_PATH}")
    df.dropna(subset=['review'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    print("=" * 60)
    print("Steam 게임 리뷰 LSTM 감성 분석 사전 계산")
    print("=" * 60)

    # ── 이전 진행 상황 확인 (resume) ─────────────────────────────
    if os.path.exists(OUTPUT_PATH):
        done_df     = pd.read_csv(OUTPUT_PATH)
        start_idx   = len(done_df)
        write_mode  = 'a'
        write_header = False
        print(f"이전 진행 이어서: {start_idx:,}건 완료")
    else:
        start_idx    = 0
        write_mode   = 'w'
        write_header = True

    # ── 데이터 로딩 ──────────────────────────────────────────────
    print("\n데이터 로딩 중...")
    df    = load_data()
    total = len(df)
    print(f"전체: {total:,}건 | 완료: {start_idx:,}건 | 남은: {total - start_idx:,}건")

    if start_idx >= total:
        print("\n이미 모든 리뷰 처리 완료!")
        print(f"결과 파일: {OUTPUT_PATH}")
        return

    # ── 모델 로딩 ────────────────────────────────────────────────
    sys.path.insert(0, BASE_DIR)
    from mylib.sentiment_analyzer import SentimentAnalyzer

    sa = SentimentAnalyzer(model_file=MODEL_PATH, tokenizer_file=TOKENIZER_PATH)
    if not sa.is_available:
        print("모델 로딩 실패. 종료합니다.")
        sys.exit(1)

    # ── 청크 단위 배치 처리 ──────────────────────────────────────
    t_start   = time.time()
    processed = 0

    with open(OUTPUT_PATH, write_mode, newline='', encoding='utf-8') as f:
        if write_header:
            f.write('idx,lstm_pos\n')

        for chunk_start in range(start_idx, total, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total)
            reviews   = df.iloc[chunk_start:chunk_end]['review'].tolist()

            preds = sa.analyze_batch(reviews)

            for i, pred in enumerate(preds):
                f.write(f'{chunk_start + i},{1 if pred == "긍정" else 0}\n')
            f.flush()

            processed += len(reviews)
            elapsed    = time.time() - t_start
            speed      = processed / elapsed if elapsed > 0 else 0
            remaining  = (total - start_idx - processed) / speed if speed > 0 else float('inf')
            pct        = (start_idx + processed) / total * 100

            print(
                f"[{start_idx + processed:,}/{total:,}]  {pct:.1f}%  "
                f"속도: {speed:.0f}건/s  남은시간: {remaining / 3600:.1f}h"
            )

    elapsed_total = time.time() - t_start
    print(f"\n완료! 총 소요 시간: {elapsed_total / 3600:.2f}시간")
    print(f"저장 위치: {OUTPUT_PATH}")
    print("\n앱을 재시작하면 사전 계산된 예측이 자동으로 사용됩니다.")


if __name__ == '__main__':
    main()
