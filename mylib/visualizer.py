from collections import Counter

import streamlit as st
import plotly.graph_objects as go


def show_summary_metrics(game_name: str, pos_count: int, neg_count: int):
    """긍정/부정 리뷰 수와 긍정 비율을 3개 메트릭 카드로 가로 배치한다."""
    total     = pos_count + neg_count
    pos_ratio = (pos_count / total * 100) if total > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("긍정 리뷰", f"{pos_count:,}건")
    with col2:
        st.metric("부정 리뷰", f"{neg_count:,}건")
    with col3:
        st.metric("긍정 비율", f"{pos_ratio:.1f}%")


def _make_barchart_plotly(counter: Counter, num_words: int, color: str) -> go.Figure:
    """Plotly 수평 막대 그래프를 생성해 반환한다. (클릭 이벤트 지원)"""
    items = counter.most_common(num_words)
    if not items:
        fig = go.Figure()
        fig.add_annotation(text="표시할 키워드가 없습니다", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    words  = [w for w, _ in reversed(items)]
    counts = [c for _, c in reversed(items)]

    fig = go.Figure(go.Bar(
        x=counts,
        y=words,
        orientation='h',
        marker_color=color,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>빈도: %{x}회<br><i>클릭하면 관련 리뷰 보기</i><extra></extra>",
        selected=dict(marker=dict(color=color, opacity=1.0)),
        unselected=dict(marker=dict(opacity=0.4)),
    ))

    fig.update_layout(
        margin=dict(l=0, r=10, t=10, b=0),
        height=max(320, len(items) * 30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Malgun Gothic, AppleGothic, sans-serif", size=12),
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', title="빈도"),
        yaxis=dict(tickfont=dict(size=12)),
        clickmode='event+select',
        dragmode=False,
    )
    return fig


@st.dialog("관련 리뷰")
def _show_review_dialog(word: str, reviews: list, polarity: str):
    """선택된 단어가 포함된 리뷰를 다이얼로그로 표시한다."""
    icon = "🟢" if polarity == "긍정" else "🔴"
    st.markdown(f"### {icon} **'{word}'** — {polarity} 리뷰 {len(reviews)}건")
    if not reviews:
        st.info("저장된 리뷰가 없습니다.")
        return
    for i, review in enumerate(reviews, 1):
        st.markdown(f"**{i}.** {review}")
        if i < len(reviews):
            st.divider()


def show_keyword_analysis(
    pos_counter: Counter,
    neg_counter: Counter,
    pos_reviews_map: dict,
    neg_reviews_map: dict,
    num_words: int,
    font_path: str = None,
):
    """긍정/부정 키워드 Plotly 막대 차트를 2열로 표시한다.

    막대 클릭 시 해당 단어가 포함된 리뷰를 다이얼로그로 표시한다.
    직전 선택값을 session_state로 추적해 두 차트가 동시에 dialog를 열지 않도록 한다.
    """
    prev_pos = st.session_state.get("_chart_sel_pos")
    prev_neg = st.session_state.get("_chart_sel_neg")

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("#### 🟢 긍정 키워드")
        fig_pos   = _make_barchart_plotly(pos_counter, num_words, "forestgreen")
        event_pos = st.plotly_chart(
            fig_pos, on_select="rerun",
            key="pos_bar_chart", use_container_width=True,
        )
        selected_pos = (
            event_pos.get("selection", {}).get("points", [{}])[0].get("y")
            if event_pos and event_pos.get("selection", {}).get("points")
            else None
        )

    with col_neg:
        st.markdown("#### 🔴 부정 키워드")
        fig_neg   = _make_barchart_plotly(neg_counter, num_words, "crimson")
        event_neg = st.plotly_chart(
            fig_neg, on_select="rerun",
            key="neg_bar_chart", use_container_width=True,
        )
        selected_neg = (
            event_neg.get("selection", {}).get("points", [{}])[0].get("y")
            if event_neg and event_neg.get("selection", {}).get("points")
            else None
        )

    # 이번 rerun에서 실제로 변경된 쪽만 dialog 오픈 (동시 오픈 방지)
    pos_changed = selected_pos is not None and selected_pos != prev_pos
    neg_changed = selected_neg is not None and selected_neg != prev_neg

    st.session_state["_chart_sel_pos"] = selected_pos
    st.session_state["_chart_sel_neg"] = selected_neg

    if pos_changed:
        _show_review_dialog(selected_pos, pos_reviews_map.get(selected_pos, []), "긍정")
    elif neg_changed:
        _show_review_dialog(selected_neg, neg_reviews_map.get(selected_neg, []), "부정")
