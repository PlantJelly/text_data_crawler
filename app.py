import os
import sys
import json
import streamlit as st

# ── 경로 설정 ───────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATA_PATH          = os.path.join(BASE_DIR, 'data', 'merged_cleaned_reviews.csv')
STOPWORDS_SAVE_PATH = os.path.join(BASE_DIR, 'data', 'custom_stopwords.json')


def _load_custom_stopwords() -> tuple:
    """(word_list, excluded_defaults) 반환. 이전 형식(list)도 호환."""
    try:
        with open(STOPWORDS_SAVE_PATH, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):          # 이전 형식 호환
            return data, []
        return data.get('word_list', []), data.get('excluded_defaults', [])
    except Exception:
        return [], []


def _save_custom_stopwords(word_list: list, excluded_defaults: list):
    with open(STOPWORDS_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump({'word_list': word_list, 'excluded_defaults': excluded_defaults},
                  f, ensure_ascii=False)

# mylib 경로 등록
sys.path.insert(0, BASE_DIR)

from mylib.game_data_loader import (
    load_game_reviews, get_all_game_names, get_game_reviews,
    analyze_keywords, DEFAULT_STOPWORDS,
    load_lstm_predictions, get_kiwi_tokenizer,
)
from mylib.visualizer import show_summary_metrics, show_keyword_analysis


# ── 페이지 설정 ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="스팀 게임 리뷰 분석",
    page_icon="🎮",
    layout="wide",
)

# ── session_state 초기화 ──────────────────────────────────────────────────
_defaults = {
    "selected_game": None,
    "is_analyzing":  False,
    "analysis_done": False,
    "last_result":   None,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 불용어는 파일에서 로드 (세션 최초 1회)
if "word_list" not in st.session_state or "excluded_defaults" not in st.session_state:
    _wl, _ex = _load_custom_stopwords()
    st.session_state.word_list        = _wl
    st.session_state.excluded_defaults = _ex


# ── 불용어 다이얼로그 ────────────────────────────────────────────────────
@st.dialog("불용어 설정")
def _stopword_dialog():
    # ── 기본 불용어 (multiselect로 개별 비활성화 가능) ──────────────
    st.caption("기본 불용어 — 선택 해제하면 분석에서 제외됩니다")
    selected = st.multiselect(
        "기본 불용어",
        options=DEFAULT_STOPWORDS,
        default=[w for w in DEFAULT_STOPWORDS if w not in st.session_state.excluded_defaults],
        label_visibility="collapsed",
    )
    new_excluded = [w for w in DEFAULT_STOPWORDS if w not in selected]
    if new_excluded != st.session_state.excluded_defaults:
        st.session_state.excluded_defaults = new_excluded
        _save_custom_stopwords(st.session_state.word_list, st.session_state.excluded_defaults)

    st.divider()

    # ── 직접 추가한 불용어 ────────────────────────────────────────
    st.caption("직접 추가한 불용어")

    def _add_word():
        w = st.session_state.get("_sw_input", "").strip()
        if w and w not in st.session_state.word_list:
            st.session_state.word_list.append(w)
            _save_custom_stopwords(st.session_state.word_list, st.session_state.excluded_defaults)
        st.session_state["_sw_input"] = ""

    st.text_input("단어 입력 후 Enter로 추가", key="_sw_input", on_change=_add_word)

    word_list = st.session_state.word_list
    if word_list:
        for i, word in enumerate(word_list):
            col1, col2 = st.columns([5, 1])
            col1.write(word)

            def _del(idx=i):
                if idx < len(st.session_state.word_list):
                    st.session_state.word_list.pop(idx)
                    _save_custom_stopwords(st.session_state.word_list, st.session_state.excluded_defaults)

            col2.button("✕", key=f"_del_{i}", on_click=_del)

        def _clear_all():
            st.session_state.word_list = []
            _save_custom_stopwords([], st.session_state.excluded_defaults)

        st.button("추가 불용어 전체 삭제", use_container_width=True, on_click=_clear_all)
    else:
        st.caption("추가된 불용어 없음")


# ── 사이드바 ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎮 게임 선택")

    # 전체 게임 목록 로드 (캐싱됨 — 최초 1회만 groupby 실행)
    all_games = get_all_game_names(DATA_PATH)
    _placeholder = "게임명을 입력하여 검색..."
    _display_opts = [_placeholder] + [f"{name}  ({cnt:,}건)" for name, cnt in all_games]

    def _on_game_change():
        st.session_state.analysis_done = False
        st.session_state.last_result   = None
        st.session_state["_chart_sel_pos"] = None
        st.session_state["_chart_sel_neg"] = None

    _sel_idx = st.selectbox(
        "게임 검색",
        options=range(len(_display_opts)),
        format_func=lambda i: _display_opts[i],
        index=0,
        key="game_selectbox",
        on_change=_on_game_change,
    )
    st.session_state.selected_game = all_games[_sel_idx - 1][0] if _sel_idx > 0 else None

    if st.session_state.selected_game:
        st.caption(f"선택됨: **{st.session_state.selected_game}**")

    st.divider()
    st.subheader("⚙️ 분석 설정")

    # 불용어 설정 버튼 — 클릭하면 팝업에서 추가/삭제 가능
    if st.button("불용어 설정", use_container_width=True):
        _stopword_dialog()

    # 키워드 수 슬라이더
    num_words = st.slider("키워드 수", min_value=10, max_value=50, value=20, step=1)

    st.divider()
    # 분석 시작 버튼
    analyze_btn = st.button("분석 시작", type="primary", use_container_width=True)


# ── 사전 계산 파일 및 형태소 분석기 로드 ────────────────────────────────────
lstm_preds, lstm_ready = load_lstm_predictions(DATA_PATH)
tokenizer = get_kiwi_tokenizer()

# ── 메인 영역 ──────────────────────────────────────────────────────────────
st.title("🎮 스팀 게임 리뷰 분석")

# 분석 전 안내
if not st.session_state.analysis_done and not st.session_state.is_analyzing:
    st.info("사이드바에서 게임명을 입력해 선택한 뒤 '분석 시작' 버튼을 클릭하세요.")


# ── 분석 시작 버튼 핸들러 ─────────────────────────────────────────────────
# 버튼 클릭 시 즉시 is_analyzing=True로 설정하고 rerun → 이전 결과가 로딩 중 보이는 문제 방지
if analyze_btn:
    if not st.session_state.selected_game:
        st.error("분석할 게임을 먼저 선택해주세요.")
    else:
        st.session_state.is_analyzing  = True
        st.session_state.analysis_done = False
        st.session_state.last_result   = None
        st.session_state["_chart_sel_pos"] = None
        st.session_state["_chart_sel_neg"] = None
        st.rerun()

# ── 분석 실행 (is_analyzing 플래그로 제어) ────────────────────────────────
if st.session_state.is_analyzing:
    try:
        with st.spinner("데이터를 분석 중입니다..."):
            df        = load_game_reviews(DATA_PATH)
            game_name = st.session_state.selected_game

            # 긍정/부정 분리 — LSTM 사전 계산 파일 사용 (없으면 voted_up 폴백)
            if lstm_ready:
                pos_reviews, neg_reviews, total, pos_idx, neg_idx = get_game_reviews(
                    df, game_name, lstm_preds=lstm_preds
                )
            else:
                st.warning("LSTM 사전 계산 파일이 없습니다. voted_up 기준으로 분류합니다.")
                pos_reviews, neg_reviews, total, pos_idx, neg_idx = get_game_reviews(df, game_name)

            my_tags = ('Noun', 'Adjective')
            active_defaults = [w for w in DEFAULT_STOPWORDS
                               if w not in st.session_state.excluded_defaults]
            my_stop = tuple(active_defaults + st.session_state.word_list)

            pos_counter, pos_reviews_map = analyze_keywords(
                tuple(pos_reviews), tokenizer,
                tags_tuple=my_tags, stopwords_tuple=my_stop,
            )
            neg_counter, neg_reviews_map = analyze_keywords(
                tuple(neg_reviews), tokenizer,
                tags_tuple=my_tags, stopwords_tuple=my_stop,
            )

            genre_series = df[df['game'] == game_name]['genre']
            game_genre   = genre_series.iloc[0] if not genre_series.empty else "N/A"

            st.session_state.is_analyzing  = False
            st.session_state.analysis_done = True
            st.session_state.last_result   = {
                "game_name":       game_name,
                "game_genre":      game_genre,
                "pos_count":       len(pos_reviews),
                "neg_count":       len(neg_reviews),
                "pos_counter":     pos_counter,
                "neg_counter":     neg_counter,
                "pos_reviews_map": pos_reviews_map,
                "neg_reviews_map": neg_reviews_map,
            }

    except Exception as e:
        st.session_state.is_analyzing = False
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        import traceback
        st.expander("상세 오류").write(traceback.format_exc())


# ── 결과 표시 ─────────────────────────────────────────────────────────────
if st.session_state.analysis_done and st.session_state.last_result:
    r = st.session_state.last_result

    # 게임명 + 장르 헤더
    st.header(f"📊 {r['game_name']}  |  {r['game_genre']}")

    # 리뷰 수 부족 경고
    if r['pos_count'] < 10:
        st.warning(f"긍정 리뷰가 {r['pos_count']}건으로 분석 결과가 부정확할 수 있습니다.")
    if r['neg_count'] < 10:
        st.warning(f"부정 리뷰가 {r['neg_count']}건으로 분석 결과가 부정확할 수 있습니다.")
    # 요약 지표
    show_summary_metrics(r['game_name'], r['pos_count'], r['neg_count'])

    st.divider()

    # 긍정/부정 키워드 시각화 (막대 클릭 → 리뷰 다이얼로그)
    show_keyword_analysis(
        r['pos_counter'], r['neg_counter'],
        r.get('pos_reviews_map', {}), r.get('neg_reviews_map', {}),
        num_words,
    )
