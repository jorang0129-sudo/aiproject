


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="월별 평균기온 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 평균기온 예측")
st.write(
    "과거 기온 데이터를 분석하고 선형회귀 알고리즘을 이용해 "
    "미래의 평균기온을 예측해 보세요!"
)

st.divider()


# ==========================================
# 2. CSV 파일 불러오기
# ==========================================
@st.cache_data
def load_data():

    # 현재 파이썬 파일은 pages 폴더에 있음
    # CSV 파일은 pages 폴더의 상위 폴더에 있음
    base_dir = Path(__file__).resolve().parent.parent

    csv_path = base_dir / "temperature.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다.\n경로: {csv_path}"
        )

    # CSV 파일 인코딩 처리
    try:
        data = pd.read_csv(csv_path, encoding="cp949")
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 필요한 열 확인
    required_columns = ["일시", "평균기온(℃)"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"CSV 파일에 '{column}' 열이 없습니다."
            )

    # 날짜 변환 (예: Oct-07 → 2007년 10월)
    data["날짜"] = pd.to_datetime(
        data["일시"],
        format="%b-%y",
        errors="coerce"
    )

    data["연도"] = data["날짜"].dt.year
    data["월"] = data["날짜"].dt.month

    # 평균기온 숫자 변환
    data["평균기온(℃)"] = pd.to_numeric(
        data["평균기온(℃)"],
        errors="coerce"
    )

    # 결측치 제거
    data = data.dropna(
        subset=["연도", "월", "평균기온(℃)"]
    ).copy()

    data["연도"] = data["연도"].astype(int)
    data["월"] = data["월"].astype(int)

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


# ==========================================
# 3. 데이터 기본 정보
# ==========================================
st.subheader("📁 기온 데이터 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터 개수", f"{len(df):,}개")

with col2:
    st.metric("데이터 시작 연도", f"{df['연도'].min()}년")

with col3:
    st.metric("데이터 마지막 연도", f"{df['연도'].max()}년")


# ==========================================
# 4. 사용자 입력
# ==========================================
st.divider()
st.subheader("📌 예측 조건 선택")

col1, col2 = st.columns(2)

with col1:
    selected_month = st.selectbox(
        "예측할 월을 선택하세요.",
        range(1, 13),
        format_func=lambda x: f"{x}월"
    )

with col2:
    last_year = int(df["연도"].max())

    prediction_year = st.number_input(
        "예측하고 싶은 연도를 입력하세요.",
        min_value=last_year + 1,
        max_value=2200,
        value=last_year + 1,
        step=1
    )

prediction_year = int(prediction_year)


# ==========================================
# 5. 선택한 월의 데이터 추출
# ==========================================
month_data = df[
    df["월"] == selected_month
].copy()

month_data = month_data.sort_values("연도")

if len(month_data) < 2:
    st.error("선택한 월의 데이터가 부족합니다.")
    st.stop()

# 독립변수 X = 연도
# 종속변수 Y = 평균기온
X = month_data["연도"].values.astype(float)
Y = month_data["평균기온(℃)"].values.astype(float)


# ==========================================
# 6. 선형회귀 알고리즘
# ==========================================

# X와 Y의 평균
mean_x = sum(X) / len(X)
mean_y = sum(Y) / len(Y)

# 기울기 계산
numerator = sum(
    (X[i] - mean_x) * (Y[i] - mean_y)
    for i in range(len(X))
)

denominator = sum(
    (X[i] - mean_x) ** 2
    for i in range(len(X))
)

if denominator == 0:
    st.error("선형회귀 분석을 진행할 수 없습니다.")
    st.stop()

slope = numerator / denominator

# y절편 계산
intercept = mean_y - slope * mean_x


# 선형회귀 예측 함수
def predict_temperature(year):
    return slope * year + intercept


# 입력한 연도의 예측 기온
predicted_temp = predict_temperature(prediction_year)

# 과거 데이터에 대한 회귀선 예측값
fitted_values = [
    predict_temperature(year) for year in X
]


# ==========================================
# 7. 결정계수 R² 계산
# ==========================================
ss_res = sum(
    (Y[i] - fitted_values[i]) ** 2
    for i in range(len(Y))
)

ss_tot = sum(
    (Y[i] - mean_y) ** 2
    for i in range(len(Y))
)

r_squared = (
    1 - ss_res / ss_tot
    if ss_tot != 0 else 0
)


# ==========================================
# 8. 예측 결과 출력
# ==========================================
st.divider()
st.subheader(f"📊 {selected_month}월 평균기온 예측 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "분석한 데이터",
        f"{len(month_data)}개"
    )

with col2:
    st.metric(
        "과거 평균기온",
        f"{mean_y:.2f}℃"
    )

with col3:
    st.metric(
        f"{prediction_year}년 예측 기온",
        f"{predicted_temp:.2f}℃"
    )

st.success(
    f"🌡️ {prediction_year}년 {selected_month}월의 "
    f"예측 평균기온은 {predicted_temp:.2f}℃입니다."
)

st.info(
    f"회귀식: y = {slope:.4f}x + {intercept:.2f} "
    f"| 결정계수 R² = {r_squared:.4f}"
)


# ==========================================
# 9. 그래프 그리기
# ==========================================
st.divider()
st.subheader("📈 실제 기온과 미래 예측 그래프")

fig, ax = plt.subplots(figsize=(13, 6))

# 실제 관측값
ax.scatter(
    X,
    Y,
    color="royalblue",
    s=45,
    label="Actual Temperature",
    zorder=3
)

# 실제 기온 연결선
ax.plot(
    X,
    Y,
    color="skyblue",
    linewidth=1.5,
    alpha=0.7,
    label="Actual Temperature Trend"
)

# 회귀선의 연도 범위
start_year = int(min(X))
end_year = prediction_year

line_years = list(
    range(start_year, end_year + 1)
)

line_temps = [
    predict_temperature(year)
    for year in line_years
]

# 선형회귀 추세선
ax.plot(
    line_years,
    line_temps,
    color="red",
    linewidth=2.5,
    linestyle="--",
    label="Linear Regression Line",
    zorder=2
)

# 잔차 표시
ax.vlines(
    X,
    fitted_values,
    Y,
    color="gray",
    alpha=0.4,
    linewidth=0.8,
    label="Residuals"
)

# 미래 예측값 표시
ax.scatter(
    prediction_year,
    predicted_temp,
    color="red",
    marker="*",
    s=350,
    edgecolors="black",
    linewidths=0.8,
    label="Predicted Temperature",
    zorder=5
)

# 예측값 설명
ax.annotate(
    f"{prediction_year}년 예측\n{predicted_temp:.2f}℃",
    xy=(prediction_year, predicted_temp),
    xytext=(10, 15),
    textcoords="offset points",
    fontsize=11,
    color="red",
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->",
        color="red"
    )
)

# 그래프 제목과 축
ax.set_title(
    f"{selected_month}월 평균기온 선형회귀 예측",
    fontsize=16
)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Average Temperature (℃)", fontsize=12)

ax.grid(True, alpha=0.3)
ax.legend(loc="best")

ax.set_xlim(
    start_year - 1,
    prediction_year + 2
)

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# ==========================================
# 10. 선형회귀 알고리즘 설명
# ==========================================
st.divider()

st.subheader("🧠 선형회귀 알고리즘이란?")

st.write("""
선형회귀(Linear Regression)는 과거에 수집한 데이터를 바탕으로
두 변수 사이의 관계를 직선으로 표현하고 미래의 값을 예측하는
머신러닝 알고리즘입니다.

이번 프로그램에서는 과거의 연도별 평균기온을 분석하여
미래의 평균기온을 예측합니다.
""")

st.markdown("### ① 선형회귀의 기본 원리")

st.latex(r"y = ax + b")

st.write("""
- **x** : 연도
- **y** : 예측한 평균기온
- **a** : 기울기
- **b** : y절편

기울기가 양수라면 시간이 지날수록 평균기온이 상승하는 경향을,
음수라면 평균기온이 하락하는 경향을 의미합니다.
""")

st.markdown("### ② 가장 적합한 직선은 어떻게 찾을까?")

st.write("""
선형회귀는 실제 관측값과 직선이 예측한 값의 차이인
잔차(Residual)를 계산합니다.

이 잔차들을 각각 제곱한 뒤 모두 더한 값이 가장 작아지도록
기울기와 절편을 결정합니다.

이 방법을 **최소제곱법(Least Squares Method)**이라고 합니다.
""")

st.latex(
    r"\text{잔차} = \text{실제 기온} - \text{예측 기온}"
)

st.latex(
    r"\text{최소화: } \sum_{i=1}^{n}(y_i-\hat{y}_i)^2"
)

st.markdown("### ③ 이번 분석의 회귀식")

st.latex(
    rf"y = {slope:.4f}x + {intercept:.2f}"
)

st.write(
    f"연도가 1년 증가할 때 평균기온은 "
    f"약 {slope:.4f}℃ 변화하는 경향이 있습니다."
)

st.markdown("### ④ 결정계수 R²란?")

st.metric(
    "이번 분석의 결정계수",
    f"{r_squared:.4f}"
)

st.write("""
결정계수는 회귀모델이 과거 기온 데이터의 변동을
얼마나 설명하는지 나타내는 지표입니다.

일반적으로 1에 가까울수록 과거 데이터에 대한 설명력이 높습니다.
단, 결정계수가 높다고 미래 예측이 반드시 정확한 것은 아닙니다.
""")

st.markdown("### ⑤ 그래프 읽는 방법")

st.markdown("""
| 그래프 표시 | 의미 |
|---|---|
| 🔵 파란 점 | 실제 관측된 평균기온 |
| 하늘색 선 | 실제 기온을 연결한 선 |
| 🔴 빨간 점선 | 선형회귀로 계산한 추세선 |
| 회색 세로선 | 실제 기온과 예측 기온의 차이 |
| ⭐ 빨간 별 | 미래의 예측 평균기온 |
""")

st.markdown("### ⑥ 선형회귀의 장점과 한계")

st.write("""
**장점**
- 계산 원리가 비교적 간단합니다.
- 데이터의 전반적인 변화 추세를 파악할 수 있습니다.
- 과거 데이터를 이용해 미래 값을 예측할 수 있습니다.

**한계**
- 기온은 다양한 자연현상의 영향을 받습니다.
- 기온 변화가 항상 직선 형태로 나타나는 것은 아닙니다.
- 먼 미래를 예측하면 실제 기온과 차이가 커질 수 있습니다.
""")

st.warning("""
⚠️ 이 프로그램의 결과는 과거 기온의 선형적인 추세를 바탕으로 한
통계적 예측입니다. 실제 기상청의 기온 예보와는 다를 수 있습니다.
""")

st.caption(
    "🌍 과거 기온 데이터로 배우는 선형회귀와 기온 예측"
)
