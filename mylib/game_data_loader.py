import os
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from kiwipiepy import Kiwi

# kiwipiepy 태그 → Okt 스타일 태그 매핑
# NNB(의존명사: 거/것/때/듯/만 등)는 독립 의미가 없으므로 제외
_KIWI_TO_OKT = {
    'NNG': 'Noun', 'NNP': 'Noun',
    'VA':  'Adjective',
    'VV':  'Verb',
}

# 기본 불용어 목록 — 의미 없는 단어 제거용
DEFAULT_STOPWORDS = [
    # 일반 불용어
    '게임', '이', '그', '저', '것', '수', '를', '을', '은', '는',
    '나', '내', '제', '우리', '좀', '더', '도',
    '근데', '그냥', '진짜', '정말', '너무', '매우', '많이',
    # Okt 활용형 (하위 호환)
    '이다', '있다', '하다', '되다', '않다', '없다', '같다', '보다',
    # kiwipiepy 어간 형태 (kiwipiepy는 어간을 분리해서 반환)
    '있', '없', '되', '하', '않', '같', '보',
    '안', '못', '다', '한',
]


@st.cache_resource
def get_kiwi_tokenizer():
    """Kiwi 인스턴스를 초기화하고 Okt 호환 형태소 분석 함수를 반환한다.
    @st.cache_resource 로 앱 전체에서 단일 인스턴스를 공유한다.

    NNG/NNP/XR + XSN 패턴은 합성명사로 합친다.
    예) 최적(NNG) + 화(XSN) → 최적화, 가능(NNG) + 성(XSN) → 가능성
    """
    kiwi = Kiwi()

    def _tokenize(text: str):
        tokens = kiwi.tokenize(text)
        result = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            # 명사/어근 + 명사파생접미사 → 합성명사로 합침
            if (t.tag in ('NNG', 'NNP', 'XR')
                    and i + 1 < len(tokens)
                    and tokens[i + 1].tag == 'XSN'):
                result.append((t.form + tokens[i + 1].form, 'Noun'))
                i += 2
            elif t.tag in _KIWI_TO_OKT:
                result.append((t.form, _KIWI_TO_OKT[t.tag]))
                i += 1
            else:
                i += 1
        return result

    return _tokenize


@st.cache_data
def load_game_reviews(csv_path: str) -> pd.DataFrame:
    """CSV 파일에서 게임 리뷰 데이터를 로드한다. 인코딩을 순서대로 시도한다."""
    df = None
    for enc in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"CSV 파일을 읽을 수 없습니다: {csv_path}")

    # review 컬럼 결측치 제거
    df.dropna(subset=['review'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # voted_up 컬럼이 문자열 'True'/'False'인 경우 불리언으로 변환
    if df['voted_up'].dtype == object:
        df['voted_up'] = df['voted_up'].map({'True': True, 'False': False})

    return df


@st.cache_data
def get_all_game_names(csv_path: str) -> list:
    """전체 게임 목록을 리뷰 수 내림차순으로 반환한다. (자동완성 selectbox용)

    반환값: [(game_name, review_count), ...]
    """
    df = load_game_reviews(csv_path)
    counts = df.groupby('game').size().sort_values(ascending=False)
    return [(name, int(cnt)) for name, cnt in counts.items()]



@st.cache_data
def load_lstm_predictions(csv_path: str) -> tuple:
    """data/lstm_sentiment.csv 가 있고 완전히 처리된 경우 (bool 배열, True) 반환.
    없거나 미완성이면 (None, False) 반환.

    precompute_sentiment.py 실행 후 앱을 재시작해야 반영된다.
    """
    lstm_path = os.path.join(os.path.dirname(csv_path), 'lstm_sentiment.csv')
    if not os.path.exists(lstm_path):
        return None, False

    lstm_df  = pd.read_csv(lstm_path)
    main_df  = load_game_reviews(csv_path)

    if len(lstm_df) != len(main_df):
        return None, False  # 사전 계산 미완성

    return lstm_df['lstm_pos'].astype(bool).values, True


def search_games(df: pd.DataFrame, query: str) -> list:
    """game 컬럼에서 query를 포함하는 게임 목록을 리뷰 수 내림차순으로 반환한다.

    반환값: [(game_name, review_count), ...]
    """
    if not query or not query.strip():
        return []

    mask = df['game'].str.contains(query.strip(), case=False, na=False)
    matched_df = df[mask]

    if matched_df.empty:
        return []

    # 게임별 리뷰 수 집계 후 내림차순 정렬
    game_counts = matched_df.groupby('game').size().sort_values(ascending=False)
    return [(game, int(count)) for game, count in game_counts.items()]


def get_game_reviews(df: pd.DataFrame, game_name: str, lstm_preds=None) -> tuple:
    """특정 게임의 긍정/부정 리뷰와 DataFrame 내 행 인덱스를 반환한다.

    lstm_preds: load_lstm_predictions()에서 받은 bool 배열.
                제공되면 voted_up 대신 LSTM 예측으로 분류한다.
    반환값: (pos_reviews, neg_reviews, total, pos_indices, neg_indices)
    """
    game_mask = df['game'].values == game_name
    game_df   = df[game_mask]

    if lstm_preds is not None and len(lstm_preds) == len(df):
        game_indices = np.where(game_mask)[0]
        game_lstm    = lstm_preds[game_indices]
        pos_idx      = game_indices[game_lstm]
        neg_idx      = game_indices[~game_lstm]
    else:
        pos_idx = game_df[game_df['voted_up'] == True].index.values
        neg_idx = game_df[game_df['voted_up'] == False].index.values

    pos_reviews = df.iloc[pos_idx]['review'].tolist()
    neg_reviews = df.iloc[neg_idx]['review'].tolist()

    return pos_reviews, neg_reviews, len(game_df), pos_idx, neg_idx


_MAX_REVIEWS_PER_WORD = 10   # 단어별 저장할 리뷰 최대 수
_REVIEW_SNIPPET_LEN  = 300  # 리뷰 저장 시 최대 글자 수


@st.cache_data
def _tokenize_reviews(reviews_tuple: tuple, _tokenizer) -> list:
    """Kiwi 형태소 분석 결과만 캐싱한다. (느린 부분만 분리)

    reviews_tuple(게임 리뷰)이 바뀔 때만 cache miss → Kiwi 재실행.
    품사·불용어 변경 시에는 이 캐시가 유지되므로 Kiwi를 다시 돌리지 않는다.
    """
    result = []
    for text in reviews_tuple:
        if not isinstance(text, str) or not text.strip():
            result.append([])
            continue
        try:
            result.append(_tokenizer(str(text)))
        except Exception:
            result.append([])
    return result


@st.cache_data
def analyze_keywords(
    reviews_tuple: tuple,
    _tokenizer,            # _로 시작 → @st.cache_data가 해시하지 않음
    tags_tuple: tuple = None,
    stopwords_tuple: tuple = None,
) -> tuple:
    """리뷰 튜플을 형태소 분석해 (Counter, reviews_map)을 반환한다.

    reviews_tuple  : tuple(리뷰 리스트) — 캐시 키로 사용하기 위해 tuple 필수
    _tokenizer     : get_kiwi_tokenizer() 반환 함수
    tags_tuple     : 추출할 품사 tuple (None이면 모든 품사)
    stopwords_tuple: 불용어 tuple (None이면 불용어 없음)

    반환값:
        counter     : Counter({word: freq})  — 전체 반환, 상위 N개 자르기는 호출 측에서
        reviews_map : {word: [review_snippet, ...]}  — 단어별 최대 10건
    """
    # 형태소 분석: 같은 게임이면 cache hit → Kiwi 재실행 없음
    pos_tags_all = _tokenize_reviews(reviews_tuple, _tokenizer)

    all_tokens   = []
    word_reviews = {}

    for text, pos_tags in zip(reviews_tuple, pos_tags_all):
        if not pos_tags:
            continue
        try:
            if tags_tuple and stopwords_tuple:
                tokens = [w for w, t in pos_tags if t in tags_tuple and w not in stopwords_tuple and len(w) >= 2]
            elif tags_tuple:
                tokens = [w for w, t in pos_tags if t in tags_tuple and len(w) >= 2]
            elif stopwords_tuple:
                tokens = [w for w, t in pos_tags if w not in stopwords_tuple and len(w) >= 2]
            else:
                tokens = [w for w, t in pos_tags if len(w) >= 2]

            unique_tokens = list(dict.fromkeys(tokens))  # 리뷰 내 중복 제거, 순서 유지
            all_tokens.extend(unique_tokens)

            snippet = str(text)[:_REVIEW_SNIPPET_LEN]
            for w in unique_tokens:
                lst = word_reviews.setdefault(w, [])
                if len(lst) < _MAX_REVIEWS_PER_WORD:
                    lst.append(snippet)

        except Exception:
            continue

    return Counter(all_tokens), word_reviews
