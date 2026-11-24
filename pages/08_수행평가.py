

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import numpy as np

# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="서울 강수량 예측",
    page_icon="🌧️",
    layout="wide"
)

# =====================================
# 2. 하늘색 그라데이션 + 비 내리는 배경
# =====================================

st.markdown("""
<style>

/* 파스텔 하늘색 그라데이션 */
.stApp {
    background: linear-gradient(
        135deg,
        #e0f7ff 0%,
        #c4edff 35%,
        #a8ddf7 70%,
        #d6f3ff 100%
    );
    background-attachment: fixed;
}

/* 은은한 빗줄기 */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;

    background-image:
        repeating-linear-gradient(
            105deg,
            transparent 0px,
            transparent 35px,
            rgba(255,255,255,0.65) 36px,
            transparent 38px
        ),
        repeating-linear-gradient(
            105deg,
            transparent 0px,
            transparent 65px,
            rgba(95,180,220,0.25) 67px,
            transparent 69px
        );

    background-size:
        120px 160px,
        180px 200px;

    animation: rainfall 1s linear infinite;
    opacity: 0.45;
}

/* 비 애니메이션 */
@keyframes rainfall {
    from {
        background-position: 0px 0px, 30px 0px;
    }
    to {
        background-position: -15px 160px, 0px 200px;
    }
}

/* 콘텐츠 표시 */
.stMainBlockContainer,
[data-testid="stHeader"],
[data-testid="stSidebar"] {
    position: relative;
    z-index: 1;
}

/* 전체 글자색 */
.stApp,
.stApp p,
.stApp label,
.stApp li {
    color: #164b70;
}

/* 제목 */
h1 {
    color: #075985 !important;
    font-weight: 800 !important;
    text-shadow: 1px 2px 4px rgba(255,255,255,0.7);
}

h2, h3 {
    color: #075985 !important;
}

/* 입력 상자 */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.9);
    border-radius: 12px;
}

/* 결과 카드 */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.9);
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(70,150,190,0.12);
}

/* 결과 숫자 */
div[data-testid="stMetricValue"] {
    color: #0284c7;
}

/* 구분선 */
hr {
    border-color: rgba(50,140,190,0.2);
}

/* 데이터 표 */
div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.8);
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# 3. 제목
# =====================================

st.title("🌧️ 서울 월별 강수량 예측")

st.markdown(
    "☔ 과거 강수량 데이터를 분석하고 "
    "선형회귀 알고리즘으로 미래 강수량을 예측해 봐요!"
)

st.divider()


# =====================================
# 4. CSV 파일 불러오기
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "weather.csv"


@st.cache_data
def load_data():

    df = pd.read_csv(
        CSV_PATH,
        encoding="cp949",
        header=11
    )

    # 열 이름 정리
    df.columns = [
        str(col).strip().replace("\t", "")
        for col in df.columns
    ]

    # 필요한 열 선택
    df = df[["일시", "강수량(mm)"]].copy()

    # 날짜 정리
    df["일시"] = df["일시"].astype(str).str.strip()

    # 월과 연도 분리
    date_parts = df["일시"].str.extract(
        r"^([A-Za-z]{3})-(\d{2})$"
    )

    month_map = {
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

    df["월"] = date_parts[0].map(month_map)

    year_2digit = pd.to_numeric(
        date_parts[1],
        errors="coerce"
    )

    # 70~99는 1900년대, 00~69는 2000년대로 변환
    df["연도"] = np.where(
        year_2digit >= 70,
        year_2digit + 1900,
        year_2digit + 2000
    )

    # 강수량 숫자 변환
    df["강수량(mm)"] = pd.to_numeric(
        df["강수량(mm)"],
        errors="coerce"
    )

    # 결측치 제거
    df = df.dropna(
        subset=["연도", "월", "강수량(mm)"]
    )

    df["연도"] = df["연도"].astype(int)
    df["월"] = df["월"].astype(int)

    return df


try:
    df = load_data()

except Exception as e:
    st.error(f"CSV 파일을 불러올 수 없습니다: {e}")
    st.stop()


# =====================================
# 5. 사용자 입력
# =====================================

st.subheader("🔍 예측 조건 선택")

col1, col2 = st.columns(2)

with col1:

    selected_month = st.selectbox(
        "🌙 예측할 월을 선택하세요",
        list(range(1, 13)),
        format_func=lambda x: f"{x}월",
        index=6
    )

with col2:

    predict_year = st.number_input(
        "📅 예측할 연도를 입력하세요",
        min_value=2027,
        max_value=2100,
        value=2030,
        step=1
    )

predict_year = int(predict_year)

st.divider()


# =====================================
# 6. 선택한 월 데이터 추출
# =====================================

month_df = df[
    df["월"] == selected_month
].copy()

month_df = month_df.sort_values("연도")

if len(month_df) < 2:
    st.error("예측에 필요한 데이터가 부족합니다.")
    st.stop()


# =====================================
# 7. 선형회귀 학습
# =====================================

X = month_df[["연도"]]
y = month_df["강수량(mm)"]

model = LinearRegression()

model.fit(X, y)


# =====================================
# 8. 미래 강수량 예측
# =====================================

prediction = model.predict(
    pd.DataFrame({
        "연도": [predict_year]
    })
)[0]

# 강수량은 음수가 될 수 없으므로 0으로 제한
prediction = max(0, prediction)

# 회귀식 정보
slope = model.coef_[0]
intercept = model.intercept_

# 결정계수
r2 = model.score(X, y)


# =====================================
# 9. 예측 결과 표시
# =====================================

st.subheader(
    f"🌧️ {selected_month}월 강수량 예측 결과"
)

result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:

    st.metric(
        "📅 예측 연도",
        f"{predict_year}년"
    )

with result_col2:

    st.metric(
        "☔ 예측 강수량",
        f"{prediction:,.1f} mm"
    )

with result_col3:

    st.metric(
        "📊 학습 데이터 수",
        f"{len(month_df)}개"
    )


st.divider()


# =====================================
# 10. 막대그래프 + 선형회귀 추세선
# =====================================

st.subheader("📊 연도별 강수량 및 예측 그래프")

fig, ax = plt.subplots(figsize=(14, 6))

# 그래프 배경
fig.patch.set_facecolor("#eaf6ff")
ax.set_facecolor("#f5fbff")

# 실제 강수량 막대그래프
ax.bar(
    month_df["연도"],
    month_df["강수량(mm)"],
    color="#54b9e8",
    edgecolor="#1675ad",
    alpha=0.85,
    width=0.7,
    label="Actual Rainfall",
    zorder=2
)

# 회귀 추세선
start_year = int(month_df["연도"].min())
end_year = max(
    int(month_df["연도"].max()),
    predict_year
)

line_years = np.linspace(
    start_year,
    end_year,
    400
)

line_df = pd.DataFrame({
    "연도": line_years
})

line_predictions = model.predict(line_df)

# 추세선의 음수 부분은 0으로 표시
line_predictions = np.maximum(line_predictions, 0)

ax.plot(
    line_years,
    line_predictions,
    color="#e63946",
    linewidth=3,
    label="Linear Regression",
    zorder=4
)

# 미래 예측 막대
ax.bar(
    predict_year,
    prediction,
    color="#ffad42",
    edgecolor="#d97706",
    linewidth=2,
    width=0.7,
    label="Predicted Rainfall",
    zorder=5
)

# 예측값 표시
ax.annotate(
    f"{prediction:.1f} mm",
    xy=(predict_year, prediction),
    xytext=(0, 12),
    textcoords="offset points",
    ha="center",
    fontsize=12,
    color="#c45b00",
    fontweight="bold"
)

# 예측 연도 구분선
ax.axvline(
    predict_year,
    color="#e67e22",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8
)

# 그래프 제목
ax.set_title(
    f"Seoul {selected_month}월 Rainfall Prediction",
    fontsize=17,
    fontweight="bold",
    color="#123b5d",
    pad=20
)

ax.set_xlabel(
    "Year",
    fontsize=12,
    color="#123b5d"
)

ax.set_ylabel(
    "Rainfall (mm)",
    fontsize=12,
    color="#123b5d"
)

# 글자색
ax.tick_params(
    axis="both",
    colors="#123b5d"
)

# 격자
ax.grid(
    axis="y",
    alpha=0.25,
    linestyle="--",
    zorder=1
)

# 범례
ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="white",
    edgecolor="#b5d8ed"
)

# 축 범위
ax.set_xlim(
    start_year - 1,
    predict_year + 2
)

ax.set_ylim(bottom=0)

fig.tight_layout()

st.pyplot(fig)

plt.close(fig)


# =====================================
# 11. 그래프 설명
# =====================================

st.markdown("""
### 🎨 그래프 읽는 방법

- 🔵 **파란색 막대:** 과거에 실제로 관측된 강수량
- 🔴 **빨간색 선:** 선형회귀 알고리즘이 계산한 추세선
- 🟠 **주황색 막대:** 사용자가 선택한 미래 연도의 예측 강수량

빨간색 선은 과거 데이터의 전체적인 경향을 나타내요.
주황색 막대는 이 직선을 이용해 계산한 미래 예측 결과예요.
""")

st.divider()


# =====================================
# 12. 선형회귀 알고리즘 설명
# =====================================

st.subheader("🧠 선형회귀 알고리즘 알아보기")

st.markdown("""
선형회귀는 과거 데이터의 관계를 가장 잘 설명하는
직선을 찾아 미래의 값을 예측하는 알고리즘이에요.
""")

st.latex(
    r"y = ax + b"
)

st.markdown(f"""
#### 📐 선형회귀 계산 결과

| 항목 | 결과 |
|---|---:|
| 기울기(a) | {slope:.4f} |
| 절편(b) | {intercept:.2f} |
| 결정계수(R²) | {r2:.4f} |

- **x:** 연도
- **y:** 강수량
- **a:** 직선의 기울기
- **b:** 절편
""")

st.info(
    "💡 기울기가 양수라면 과거 데이터에서 강수량이 "
    "증가하는 경향, 음수라면 감소하는 경향을 나타내요. "
    "R²는 회귀선이 과거 데이터의 변동을 얼마나 설명하는지 "
    "보여주는 값이에요."
)

st.warning(
    "선형회귀는 과거의 장기적인 추세를 바탕으로 예측합니다. "
    "강수량은 자연적인 변동이 크므로 실제 미래 강수량과 "
    "차이가 발생할 수 있어요."
)

st.divider()


# =====================================
# 13. 실제 강수량 데이터 표
# =====================================

st.subheader(
    f"📋 {selected_month}월 연도별 실제 강수량 데이터"
)

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

st.caption(
    "📍 자료: 업로드한 기상청 서울 월별 강수량 CSV 데이터"
)

# =====================================
# 14. 마무리
# =====================================

st.markdown("""
---
<div style="
    text-align:center;
    color:#28769d;
    font-size:16px;
    font-weight:bold;
">
    🌧️ 서울 강수량 데이터 분석 프로젝트 ☔
</div>
""", unsafe_allow_html=True)
