
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import re
import random

# --------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="강수량 예측 🌧️",
    page_icon="💧",
    layout="wide"
)


# --------------------------------------------------
# 2. 배경 디자인 및 랜덤 물방울 효과
# --------------------------------------------------

rain_css = """
<style>

/* 전체 배경 */
.stApp {
    background: linear-gradient(
        180deg,
        #fafdff 0%,
        #f0faff 35%,
        #eaf7ff 70%,
        #f8fdff 100%
    );
    background-attachment: fixed;
    overflow-x: hidden;
}

/* 물방울 전체 레이어 */
.rain-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
}

/* 각각의 물방울 */
.raindrop {
    position: absolute;
    top: -50px;
    animation-name: falling;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    will-change: transform;
}

/* 물방울 낙하 */
@keyframes falling {
    0% {
        transform: translateY(-60px) rotate(0deg);
    }

    100% {
        transform: translateY(110vh) rotate(15deg);
    }
}

/* 콘텐츠 */
.stMainBlockContainer {
    position: relative;
    z-index: 1;
}

/* 제목 */
h1, h2, h3 {
    color: #397da8 !important;
}

/* 일반 글자 */
p, label, span {
    color: #42677f;
}

/* 입력 상자 */
div[data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 10px;
}

</style>
"""

# 물방울 120개 생성
rain_drops = []

for i in range(120):

    left = random.uniform(0, 100)
    size = random.randint(12, 24)
    duration = random.uniform(5, 13)
    delay = random.uniform(-13, 0)
    opacity = random.uniform(0.12, 0.45)

    drop = f"""
    <span class="raindrop" style="
        left: {left:.2f}%;
        font-size: {size}px;
        opacity: {opacity:.2f};
        animation-duration: {duration:.2f}s;
        animation-delay: {delay:.2f}s;
    ">💧</span>
    """

    rain_drops.append(drop)

rain_html = (
    rain_css
    + '<div class="rain-container">'
    + "".join(rain_drops)
    + "</div>"
)

st.markdown(rain_html, unsafe_allow_html=True)


# --------------------------------------------------
# 3. 제목
# --------------------------------------------------

st.title("🌧️ 월별 강수량 예측")

st.markdown(
    "과거의 강수량 데이터를 분석해서 미래의 강수량을 예측해 봐요! 💧"
)

st.divider()


# --------------------------------------------------
# 4. CSV 파일 불러오기
# --------------------------------------------------

# pages 폴더의 상위 폴더에 있는 CSV 파일
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "weather.csv"

try:
    df = pd.read_csv(
        CSV_PATH,
        encoding="cp949",
        header=11
    )

except Exception as e:
    st.error("CSV 파일을 불러오지 못했어요.")
    st.error(f"오류 내용: {e}")
    st.stop()


# --------------------------------------------------
# 5. 데이터 전처리
# --------------------------------------------------

required_columns = ["일시", "강수량(mm)"]

if not all(col in df.columns for col in required_columns):
    st.error(
        "CSV 파일에서 '일시' 또는 '강수량(mm)' 열을 찾을 수 없어요."
    )
    st.write("현재 CSV 열 목록:", list(df.columns))
    st.stop()

# 필요한 열만 선택
df = df[["일시", "강수량(mm)"]].copy()

# 결측값 제거 및 강수량 숫자 변환
df["강수량(mm)"] = pd.to_numeric(
    df["강수량(mm)"],
    errors="coerce"
)

df = df.dropna(subset=["일시", "강수량(mm)"])


# 월 이름 변환
month_dict = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}


# 날짜에서 월과 연도 추출
def extract_date(value):

    value = str(value).strip()

    match = re.search(
        r"([A-Za-z]{3})-(\d{2})",
        value
    )

    if match:
        month_name = match.group(1).title()
        year_2digit = int(match.group(2))

        month = month_dict.get(month_name)

        if year_2digit >= 70:
            year = 1900 + year_2digit
        else:
            year = 2000 + year_2digit

        return pd.Series([year, month])

    return pd.Series([np.nan, np.nan])


df[["연도", "월"]] = df["일시"].apply(extract_date)

df = df.dropna(subset=["연도", "월"])

df["연도"] = df["연도"].astype(int)
df["월"] = df["월"].astype(int)

# 연도와 월 순서로 정렬
df = df.sort_values(["연도", "월"])


# --------------------------------------------------
# 6. 월별 데이터 선택
# --------------------------------------------------

st.subheader("📅 예측 조건 선택")

col1, col2 = st.columns(2)

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

with col1:
    selected_month = st.selectbox(
        "🌱 분석할 월을 선택하세요",
        list(month_names.keys()),
        format_func=lambda x: month_names[x]
    )

with col2:
    selected_year = st.number_input(
        "🔮 예측할 연도를 선택하세요",
        min_value=2027,
        max_value=2100,
        value=2030,
        step=1
    )


# 선택한 월의 데이터만 추출
month_df = df[df["월"] == selected_month].copy()

if len(month_df) < 2:
    st.warning("선택한 월의 데이터가 충분하지 않아요.")
    st.stop()


# --------------------------------------------------
# 7. 선형회귀 학습
# --------------------------------------------------

X = month_df[["연도"]]
y = month_df["강수량(mm)"]

model = LinearRegression()
model.fit(X, y)

# 선택한 연도의 강수량 예측
prediction = model.predict(
    pd.DataFrame({"연도": [selected_year]})
)[0]

# 강수량은 음수가 될 수 없으므로 0으로 제한
prediction = max(0, prediction)

# 회귀식 정보
slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(X, y)


# --------------------------------------------------
# 8. 예측 결과 표시
# --------------------------------------------------

st.divider()
st.subheader("☔ 강수량 예측 결과")

result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:
    st.metric(
        label="선택한 월",
        value=month_names[selected_month]
    )

with result_col2:
    st.metric(
        label="예측 연도",
        value=f"{selected_year}년"
    )

with result_col3:
    st.metric(
        label="예측 강수량",
        value=f"{prediction:.1f} mm"
    )

st.info(
    f"💧 선형회귀 알고리즘이 예측한 "
    f"{selected_year}년 {month_names[selected_month]}의 "
    f"예상 강수량은 **{prediction:.1f}mm**입니다."
)


# --------------------------------------------------
# 9. 과거 강수량과 예측 강수량 막대그래프
# --------------------------------------------------

st.divider()
st.subheader("📊 과거 강수량과 미래 예측 비교")

fig, ax = plt.subplots(figsize=(13, 6))

# 과거 데이터 막대그래프
ax.bar(
    month_df["연도"],
    month_df["강수량(mm)"],
    color="#9bd8f5",
    edgecolor="#6bbce5",
    alpha=0.85,
    label="실제 강수량"
)

# 회귀선
min_year = int(month_df["연도"].min())
max_year = max(
    int(month_df["연도"].max()),
    int(selected_year)
)

line_years = np.arange(min_year, max_year + 1)

line_predictions = model.predict(
    pd.DataFrame({"연도": line_years})
)

line_predictions = np.maximum(line_predictions, 0)

ax.plot(
    line_years,
    line_predictions,
    color="#ff8585",
    linewidth=2.5,
    label="선형회귀 예측 추세"
)

# 예측 연도 막대
ax.bar(
    selected_year,
    prediction,
    color="#ffb36b",
    edgecolor="#f28c45",
    width=0.7,
    label="예측 강수량"
)

# 예측값 표시
ax.annotate(
    f"{prediction:.1f}mm",
    xy=(selected_year, prediction),
    xytext=(0, 10),
    textcoords="offset points",
    ha="center",
    fontsize=11,
    color="#e87532",
    fontweight="bold"
)

ax.set_title(
    f"{month_names[selected_month]} 연도별 강수량 및 예측",
    fontsize=16,
    pad=20
)

ax.set_xlabel("연도", fontsize=12)
ax.set_ylabel("강수량 (mm)", fontsize=12)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 10. 선형회귀 알고리즘 설명
# --------------------------------------------------

st.divider()
st.subheader("🤖 선형회귀 알고리즘 알아보기")

st.markdown("""
**선형회귀(Linear Regression)**는 과거 데이터의 관계를 분석해서
미래의 값을 예측하는 대표적인 머신러닝 알고리즘이에요.

이번 프로그램에서는 다음과 같이 분석했어요.

1. 과거의 연도를 입력 데이터로 사용해요.
2. 각 연도의 해당 월 강수량을 정답 데이터로 사용해요.
3. 과거 데이터의 전체적인 변화 경향을 직선으로 나타내요.
4. 그 직선을 이용해 선택한 연도의 강수량을 예측해요.
""")

st.markdown("### 📐 선형회귀 계산식")

st.latex(r"y = ax + b")

st.write(f"- **a (기울기):** {slope:.4f}")

st.write(f"- **b (절편):** {intercept:.4f}")

st.write(f"- **결정계수 (R²):** {r2:.4f}")

st.caption(
    "결정계수는 모델이 과거 데이터의 변화를 얼마나 설명하는지 "
    "참고하는 값이에요. 미래 예측의 정확도를 보장하지는 않아요."
)


# --------------------------------------------------
# 11. 데이터 표 확인
# --------------------------------------------------

st.divider()
st.subheader("📋 선택한 월의 과거 강수량 데이터")

display_df = month_df[
    ["연도", "강수량(mm)"]
].copy()

display_df = display_df.sort_values(
    "연도",
    ascending=False
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 12. 마무리 안내
# --------------------------------------------------

st.divider()

st.markdown("""
<div style="
    background: rgba(255,255,255,0.7);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
">
    <h3>🌧️ 강수량 예측 완료!</h3>
    <p>과거의 날씨 데이터를 활용해 미래의 강수량을 살펴봤어요.</p>
    <p>머신러닝으로 날씨 데이터를 분석하는 방법을 알아보았어요! 💙</p>
</div>
""", unsafe_allow_html=True)
