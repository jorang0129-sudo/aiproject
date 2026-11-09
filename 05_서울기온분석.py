
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="월별 기온분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 기온분석")
st.write("원하는 월을 선택하면 연도별 최고기온과 최저기온을 확인할 수 있어요.")

# CSV 파일 경로
CSV_PATH = Path(__file__).parent / "temperature.csv"


# 데이터 불러오기
@st.cache_data
def load_data(file_path):

    # 한글 인코딩 처리
    try:
        data = pd.read_csv(file_path, encoding="cp949")
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding="utf-8-sig")

    # 날짜 변환 (예: Oct-07, Jan-08)
    data["일시"] = pd.to_datetime(
        data["일시"],
        format="%b-%y",
        errors="coerce"
    )

    # 기온 데이터 숫자로 변환
    temperature_columns = [
        "최고기온(℃)",
        "최저기온(℃)"
    ]

    for column in temperature_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # 결측치 제거
    data = data.dropna(
        subset=["일시"] + temperature_columns
    ).copy()

    # 연도와 월 추출
    data["연도"] = data["일시"].dt.year
    data["월"] = data["일시"].dt.month

    return data


# CSV 파일 확인
if not CSV_PATH.exists():
    st.error("temperature.csv 파일을 찾을 수 없습니다.")
    st.stop()


# 데이터 읽기
try:
    df = load_data(CSV_PATH)

except Exception as error:
    st.error(f"데이터를 읽는 중 오류가 발생했습니다: {error}")
    st.stop()


# 월 선택
month_names = {
    1: "1월",
    2: "2월",
    3: "3월",
    4: "4월",
    5: "5월",
    6: "6월",
    7: "7월",
    8: "8월",
    9: "9월",
    10: "10월",
    11: "11월",
    12: "12월"
}

selected_month = st.selectbox(
    "📅 분석할 월을 선택하세요.",
    options=list(month_names.keys()),
    format_func=lambda x: month_names[x]
)


# 선택한 월 데이터 추출
monthly_data = df[
    df["월"] == selected_month
].copy()

# 연도순 정렬
monthly_data = monthly_data.sort_values("연도")


# 연도별 데이터 정리
yearly_data = (
    monthly_data
    .groupby("연도", as_index=False)
    [["최고기온(℃)", "최저기온(℃)"]]
    .mean()
    .sort_values("연도")
)


# 데이터 존재 여부 확인
if yearly_data.empty:
    st.warning("선택한 월의 데이터가 없습니다.")
    st.stop()


# 그래프 생성
fig = go.Figure()


# 최고기온 그래프 (핫핑크)
fig.add_trace(
    go.Scatter(
        x=yearly_data["연도"],
        y=yearly_data["최고기온(℃)"],
        mode="lines+markers",
        name="최고기온",
        line=dict(
            color="hotpink",
            width=3
        ),
        marker=dict(
            color="hotpink",
            size=7
        )
    )
)


# 최저기온 그래프 (파란색)
fig.add_trace(
    go.Scatter(
        x=yearly_data["연도"],
        y=yearly_data["최저기온(℃)"],
        mode="lines+markers",
        name="최저기온",
        line=dict(
            color="blue",
            width=3
        ),
        marker=dict(
            color="blue",
            size=7
        )
    )
)


# 그래프 제목 및 축 설정
fig.update_layout(
    title="월별 기온분석",
    xaxis_title="연도",
    yaxis_title="온도(℃)",
    legend_title="기온 구분",
    hovermode="x unified",
    template="plotly_white"
)


# 그래프 출력
st.plotly_chart(
    fig,
    use_container_width=True
)


# 데이터 표 출력
st.subheader(
    f"📊 {selected_month}월 연도별 기온 데이터"
)

st.dataframe(
    yearly_data,
    use_container_width=True,
    hide_index=True
)
