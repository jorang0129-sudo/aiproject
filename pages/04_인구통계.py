
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ----------------------------------
# 1. 페이지 설정
# ----------------------------------
st.set_page_config(
    page_title="서울시 연령별 인구 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 서울시 연령별 인구 분석")
st.write("서울시 자치구별 연령대 인구를 그래프로 분석해 보세요!")

# CSV 파일 경로
CSV_PATH = Path(__file__).parent / "population.csv"

# 연령대 목록
AGE_GROUPS = [
    "0~9세",
    "10~19세",
    "20~29세",
    "30~39세",
    "40~49세",
    "50~59세",
    "60~69세",
    "70~79세",
    "80~89세",
    "90~99세",
    "100세 이상"
]


# ----------------------------------
# 2. CSV 파일 불러오기
# ----------------------------------
@st.cache_data
def load_data(path):

    # 한글 파일이 깨지지 않도록 CP949 적용
    df = pd.read_csv(
        path,
        encoding="cp949"
    )

    # 인구수의 쉼표 제거 및 숫자 변환
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )

    # 행정구역 이름 정리
    df["행정구역"] = (
        df["행정구역"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s*\(\d+\)\s*$", "", regex=True)
    )

    return df


# ----------------------------------
# 3. 데이터 오류 확인
# ----------------------------------
if not CSV_PATH.exists():
    st.error("population.csv 파일을 찾을 수 없습니다.")
    st.stop()

try:
    df = load_data(str(CSV_PATH))

except UnicodeDecodeError:
    st.error("CSV 파일의 한글 인코딩을 확인해 주세요.")
    st.stop()

except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


# ----------------------------------
# 4. 연도 및 행정구 목록 만들기
# ----------------------------------
years = []

for year in range(2020, 2026):

    prefix = f"{year}년_거주자_"

    if any(
        str(col).startswith(prefix)
        for col in df.columns
    ):
        years.append(str(year))

if not years:
    st.error("연도별 인구 데이터를 찾을 수 없습니다.")
    st.stop()

districts = df["행정구역"].dropna().unique().tolist()


# ----------------------------------
# 5. 선택 메뉴 만들기
# ----------------------------------
st.subheader("🔎 분석 조건 선택")

col1, col2, col3 = st.columns(3)

with col1:

    selected_district = st.selectbox(
        "행정구 선택",
        districts
    )

with col2:

    # 현재 CSV에는 성별 구분 데이터가 없음
    selected_gender = st.selectbox(
        "성별 선택",
        ["전체 인구 (성별 구분 없음)"],
        help="현재 데이터는 남녀 구분 없이 전체 인구를 제공합니다."
    )

with col3:

    selected_year = st.selectbox(
        "연도 선택",
        years,
        index=len(years) - 1,
        format_func=lambda year: f"{year}년"
    )


# ----------------------------------
# 6. 선택한 데이터 추출
# ----------------------------------
selected_rows = df[
    df["행정구역"] == selected_district
]

if selected_rows.empty:
    st.error("선택한 행정구의 데이터가 없습니다.")
    st.stop()

row = selected_rows.iloc[0]

values = []

for age in AGE_GROUPS:

    column_name = f"{selected_year}년_거주자_{age}"

    if (
        column_name in df.columns
        and pd.notna(row[column_name])
    ):
        values.append(int(row[column_name]))

    else:
        values.append(0)


# 그래프용 데이터 생성
chart_data = pd.DataFrame({
    "나이": AGE_GROUPS,
    "인구수": values
})

total = sum(values)


# ----------------------------------
# 7. 주요 인구 정보
# ----------------------------------
st.subheader("📌 선택한 지역 정보")

m1, m2, m3 = st.columns(3)

m1.metric(
    "선택한 행정구",
    selected_district
)

m2.metric(
    "선택 연도",
    f"{selected_year}년"
)

m3.metric(
    "연령별 인구 합계",
    f"{total:,}명"
)


# ----------------------------------
# 8. 꺾은선 그래프 생성
# ----------------------------------
st.subheader("📈 연령대별 인구수 그래프")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_data["나이"],
        y=chart_data["인구수"],

        mode="lines+markers",

        name="인구수",

        # 하늘색 그래프
        line=dict(
            color="#87CEEB",
            width=3
        ),

        marker=dict(
            color="#87CEEB",
            size=8
        ),

        hovertemplate=
        "나이: %{x}<br>" +
        "인구수: %{y:,.0f}명" +
        "<extra></extra>"
    )
)


# ----------------------------------
# 9. 그래프 눈금 및 디자인
# ----------------------------------

max_value = max(values) if values else 0

y_max = max(
    100,
    ((max_value // 100) + 1) * 100
)

fig.update_layout(

    title=f"{selected_district} {selected_year}년 연령대별 인구수",

    xaxis=dict(
        title="나이",

        type="category",

        categoryorder="array",

        categoryarray=AGE_GROUPS,

        tickmode="array",

        tickvals=AGE_GROUPS,

        ticktext=AGE_GROUPS,

        showgrid=True,

        dtick=1,

        gridcolor="#DCEAF2",

        gridwidth=1
    ),

    yaxis=dict(
        title="인구수 (명)",

        range=[0, y_max],

        tickmode="linear",

        tick0=0,

        # 세로축 100명 단위
        dtick=100,

        tickformat=",",

        showgrid=True,

        gridcolor="#DCEAF2",

        gridwidth=1
    ),

    plot_bgcolor="white",

    paper_bgcolor="white",

    font=dict(
        family="Malgun Gothic, Apple SD Gothic Neo, NanumGothic, sans-serif",
        size=12,
        color="#263238"
    ),

    hovermode="x unified",

    height=550,

    margin=dict(
        l=80,
        r=30,
        t=80,
        b=80
    )
)


# ----------------------------------
# 10. 그래프 화면에 출력
# ----------------------------------
st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "※ 가로축은 10세 단위이며, "
    "세로축 눈금과 격자는 100명 단위입니다."
)


# ----------------------------------
# 11. 인구 데이터 표 출력
# ----------------------------------
with st.expander("📋 인구 데이터 자세히 보기"):

    table = chart_data.copy()

    table["인구수"] = table["인구수"].map(
        lambda n: f"{n:,}명"
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


# ----------------------------------
# 12. 성별 데이터 안내
# ----------------------------------
st.info(
    "현재 population.csv에는 남성과 여성의 "
    "인구를 구분하는 데이터가 없습니다. "
    "따라서 전체 인구만 표시됩니다."
)

st.success("🎉 서울시 인구 분석이 완료되었습니다!")
