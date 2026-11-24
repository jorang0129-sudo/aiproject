
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="서울 강수량 예측",
    page_icon="🌧️",
    layout="wide"
)

# 제목
st.title("🌧️ 서울 월별 강수량 예측")
st.markdown("과거 강수량 데이터를 분석하고 선형회귀로 미래 강수량을 예측해 봐요!")
st.divider()

# CSV 파일 경로
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "weather.csv"

# 데이터 불러오기
@st.cache_data
def load_data():

    # 기상청 CSV는 CP949 인코딩이며,
    # 12번째 줄부터 실제 데이터 표가 시작됨
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

    # 연도와 월 추출
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3,
        "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9,
        "Oct": 10, "Nov": 11, "Dec": 12
    }

    df["일시"] = df["일시"].astype(str).str.strip()

    df["월"] = df["일시"].str[:3].map(month_map)

    df["연도"] = pd.to_numeric(
        df["일시"].str[-2:],
        errors="coerce"
    )

    # 2자리 연도를 4자리 연도로 변환
    df["연도"] = df["연도"].apply(
        lambda x: int(x + 1900) if x >= 70
        else int(x + 2000) if pd.notna(x)
        else None
    )

    # 강수량 숫자로 변환
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
    st.error(f"CSV 파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# 사용자 입력
st.subheader("🔍 예측 조건 선택")

col1, col2 = st.columns(2)

with col1:
    selected_month = st.selectbox(
        "예측할 월을 선택하세요",
        list(range(1, 13)),
        format_func=lambda x: f"{x}월",
        index=6
    )

with col2:
    predict_year = st.number_input(
        "예측할 연도를 입력하세요",
        min_value=2027,
        max_value=2100,
        value=2030,
        step=1
    )

st.divider()


# 선택한 월의 데이터 추출
month_df = df[df["월"] == selected_month].copy()

month_df = month_df.sort_values("연도")

if len(month_df) < 2:
    st.error("예측에 필요한 데이터가 부족합니다.")
    st.stop()


# 선형회귀 학습
X = month_df[["연도"]]
y = month_df["강수량(mm)"]

model = LinearRegression()
model.fit(X, y)


# 예측
prediction = model.predict(
    pd.DataFrame({"연도": [predict_year]})
)[0]

# 강수량은 음수가 될 수 없으므로 0으로 제한
prediction = max(0, prediction)


# 회귀식 정보
slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(X, y)


# 결과 표시
st.subheader(f"🌧️ {selected_month}월 강수량 예측 결과")

result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:
    st.metric(
        "예측 연도",
        f"{predict_year}년"
    )

with result_col2:
    st.metric(
        "예측 강수량",
        f"{prediction:,.1f} mm"
    )

with result_col3:
    st.metric(
        "학습 데이터 수",
        f"{len(month_df)}개"
    )


# 그래프
st.subheader("📈 과거 강수량과 선형회귀 예측 그래프")

fig, ax = plt.subplots(figsize=(12, 6))

# 실제 강수량 산점도
ax.scatter(
    month_df["연도"],
    month_df["강수량(mm)"],
    color="royalblue",
    alpha=0.75,
    s=45,
    label="Actual Rainfall"
)

# 회귀선 생성
start_year = min(month_df["연도"].min(), predict_year)
end_year = max(month_df["연도"].max(), predict_year)

line_years = np.linspace(
    start_year,
    end_year,
    300
)

line_df = pd.DataFrame({
    "연도": line_years
})

line_predictions = model.predict(line_df)

# 회귀 추세선
ax.plot(
    line_years,
    line_predictions,
    color="red",
    linewidth=2.5,
    label="Linear Regression"
)

# 예측 지점
ax.scatter(
    predict_year,
    prediction,
    color="darkorange",
    edgecolors="black",
    s=180,
    marker="*",
    zorder=5,
    label="Predicted Rainfall"
)

# 예측 연도 표시
ax.axvline(
    predict_year,
    color="gray",
    linestyle="--",
    alpha=0.6
)

ax.annotate(
    f"{prediction:.1f} mm",
    xy=(predict_year, prediction),
    xytext=(10, 15),
    textcoords="offset points",
    fontsize=11,
    color="darkorange",
    fontweight="bold"
)

# 그래프 꾸미기
ax.set_title(
    f"Seoul {selected_month}월 Rainfall Prediction",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Rainfall (mm)", fontsize=12)

ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.divider()


# 선형회귀 알고리즘 설명
st.subheader("🧠 선형회귀 알고리즘 알아보기")

st.markdown("""
선형회귀는 과거 데이터의 분포를 가장 잘 설명하는
하나의 직선을 찾아 미래의 값을 예측하는 알고리즘이에요.
""")

st.latex(
    r"y = ax + b"
)

st.markdown(f"""
- **x:** 연도
- **y:** 강수량
- **a (기울기):** {slope:.4f}
- **b (절편):** {intercept:.2f}
- **R² 결정계수:** {r2:.4f}

빨간색 직선은 과거 강수량 데이터에 가장 잘 맞도록
선형회귀가 계산한 추세선이에요.

파란색 점은 실제 관측된 강수량이고,
주황색 별은 선택한 연도의 예측 강수량이에요.
""")

st.info(
    "R²는 회귀선이 과거 데이터의 변동을 얼마나 설명하는지 "
    "나타내는 값이에요. 1에 가까울수록 설명력이 높습니다. "
    "다만 미래 예측이 정확하다는 보장은 아니에요."
)

st.divider()


# 과거 데이터 표
st.subheader(f"📋 {selected_month}월 연도별 실제 강수량")

display_df = month_df[["연도", "강수량(mm)"]].copy()

display_df.columns = ["연도", "강수량(mm)"]

st.dataframe(
    display_df.sort_values("연도", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "자료: 업로드한 기상청 서울 월별 강수량 CSV 데이터"
)

