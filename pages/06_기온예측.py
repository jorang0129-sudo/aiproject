
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="월별 평균기온 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 평균기온 예측 프로그램")
st.write(
    "과거 기온 데이터를 바탕으로 3가지 회귀 알고리즘을 학습하고 "
    "미래의 평균기온을 예측해 보세요!"
)

st.divider()


# ==========================================
# 2. CSV 파일 불러오기
# ==========================================
@st.cache_data
def load_data():

    # pages 폴더의 상위 폴더에 있는 CSV 파일
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "temperature.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"temperature.csv 파일을 찾을 수 없습니다.\n{csv_path}"
        )

    try:
        data = pd.read_csv(csv_path, encoding="cp949")
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_columns = ["일시", "평균기온(℃)"]

    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"CSV에 '{col}' 열이 없습니다.")

    # 날짜 변환
    data["날짜"] = pd.to_datetime(
        data["일시"],
        format="%b-%y",
        errors="coerce"
    )

    data["연도"] = data["날짜"].dt.year
    data["월"] = data["날짜"].dt.month

    data["평균기온(℃)"] = pd.to_numeric(
        data["평균기온(℃)"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["연도", "월", "평균기온(℃)"]
    ).copy()

    data["연도"] = data["연도"].astype(int)
    data["월"] = data["월"].astype(int)

    # 같은 월과 연도에 데이터가 여러 개 있으면 평균 처리
    data = (
        data.groupby(["연도", "월"], as_index=False)
        ["평균기온(℃)"].mean()
    )

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


# ==========================================
# 3. 데이터 정보
# ==========================================
st.subheader("📁 기온 데이터 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터", f"{len(df):,}개")

with col2:
    st.metric("시작 연도", f"{df['연도'].min()}년")

with col3:
    st.metric("마지막 연도", f"{df['연도'].max()}년")


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
# 5. 알고리즘 선택
# ==========================================
st.divider()
st.subheader("🤖 회귀 알고리즘 선택")

algorithm = st.radio(
    "사용할 예측 알고리즘을 선택하세요.",
    [
        "선형회귀",
        "다항회귀",
        "의사결정나무 회귀"
    ],
    horizontal=True
)

st.caption(
    "알고리즘을 변경하면 그래프와 선택 알고리즘의 예측 결과가 바뀝니다."
)


# ==========================================
# 6. 선택한 월의 데이터 추출
# ==========================================
month_data = df[
    df["월"] == selected_month
].copy()

month_data = month_data.sort_values("연도")

if len(month_data) < 3:
    st.error("회귀분석에 필요한 데이터가 부족합니다.")
    st.stop()

X = month_data["연도"].values.astype(float)
Y = month_data["평균기온(℃)"].values.astype(float)

# 연도는 실제 연도가 아닌 중심화된 값으로도 사용
# 수치 계산의 안정성을 높이기 위한 처리
center_year = float(X.mean())
scale_year = float(X.max() - X.min())

if scale_year == 0:
    st.error("연도 데이터가 부족합니다.")
    st.stop()

Z = (X - center_year) / scale_year
prediction_z = (prediction_year - center_year) / scale_year


# ==========================================
# 7. 공통 함수
# ==========================================

# 결정계수 R²
def calculate_r2(actual, predicted):

    mean_y = sum(actual) / len(actual)

    ss_res = sum(
        (actual[i] - predicted[i]) ** 2
        for i in range(len(actual))
    )

    ss_tot = sum(
        (actual[i] - mean_y) ** 2
        for i in range(len(actual))
    )

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot


# 3차 연립방정식까지 풀 수 있는 가우스 소거법
def solve_linear_system(matrix, vector):

    n = len(vector)

    a = [
        list(matrix[i]) + [vector[i]]
        for i in range(n)
    ]

    for i in range(n):

        # 피벗 선택
        pivot = max(
            range(i, n),
            key=lambda r: abs(a[r][i])
        )

        if abs(a[pivot][i]) < 1e-12:
            raise ValueError("회귀계수를 계산할 수 없습니다.")

        a[i], a[pivot] = a[pivot], a[i]

        # 피벗 행 정규화
        divisor = a[i][i]

        for j in range(i, n + 1):
            a[i][j] /= divisor

        # 다른 행 제거
        for k in range(n):

            if k == i:
                continue

            factor = a[k][i]

            for j in range(i, n + 1):
                a[k][j] -= factor * a[i][j]

    return [a[i][n] for i in range(n)]


# ==========================================
# 8. 선형회귀
# ==========================================
def fit_linear(x, y):

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum(
        (x[i] - mean_x) * (y[i] - mean_y)
        for i in range(len(x))
    )

    denominator = sum(
        (x[i] - mean_x) ** 2
        for i in range(len(x))
    )

    if denominator == 0:
        raise ValueError("선형회귀 계산에 실패했습니다.")

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    def predict(year):
        return slope * year + intercept

    return predict, slope, intercept


# ==========================================
# 9. 다항회귀 (2차)
# ==========================================
def fit_polynomial(x, y):

    # z를 이용해 연도 데이터를 정규화
    # y = c0 + c1*z + c2*z^2
    n = len(x)

    s0 = n
    s1 = sum(x)
    s2 = sum(v ** 2 for v in x)
    s3 = sum(v ** 3 for v in x)
    s4 = sum(v ** 4 for v in x)

    t0 = sum(y)
    t1 = sum(x[i] * y[i] for i in range(n))
    t2 = sum((x[i] ** 2) * y[i] for i in range(n))

    matrix = [
        [s0, s1, s2],
        [s1, s2, s3],
        [s2, s3, s4]
    ]

    vector = [t0, t1, t2]

    c0, c1, c2 = solve_linear_system(
        matrix,
        vector
    )

    def predict(year):
        return c0 + c1 * year + c2 * year ** 2

    return predict, c0, c1, c2


# ==========================================
# 10. 의사결정나무 회귀
# ==========================================
def fit_decision_tree(x, y, max_depth=3, min_leaf=2):

    # 각 노드의 예측값은 해당 구간의 평균기온
    def build_tree(xs, ys, depth):

        mean_value = sum(ys) / len(ys)

        node = {
            "value": mean_value,
            "left": None,
            "right": None,
            "threshold": None
        }

        # 종료 조건
        if depth >= max_depth:
            return node

        if len(xs) < min_leaf * 2:
            return node

        best_sse = float("inf")
        best_threshold = None

        # 모든 가능한 분할 지점 탐색
        for i in range(min_leaf, len(xs) - min_leaf + 1):

            if xs[i - 1] >= xs[i]:
                continue

            left_y = ys[:i]
            right_y = ys[i:]

            left_mean = sum(left_y) / len(left_y)
            right_mean = sum(right_y) / len(right_y)

            left_sse = sum(
                (v - left_mean) ** 2 for v in left_y
            )

            right_sse = sum(
                (v - right_mean) ** 2 for v in right_y
            )

            total_sse = left_sse + right_sse

            if total_sse < best_sse:
                best_sse = total_sse
                best_threshold = (xs[i - 1] + xs[i]) / 2

        if best_threshold is None:
            return node

        left_indices = [
            i for i in range(len(xs))
            if xs[i] <= best_threshold
        ]

        right_indices = [
            i for i in range(len(xs))
            if xs[i] > best_threshold
        ]

        node["threshold"] = best_threshold

        node["left"] = build_tree(
            [xs[i] for i in left_indices],
            [ys[i] for i in left_indices],
            depth + 1
        )

        node["right"] = build_tree(
            [xs[i] for i in right_indices],
            [ys[i] for i in right_indices],
            depth + 1
        )

        return node

    def predict_one(node, value):

        if node["threshold"] is None:
            return node["value"]

        if value <= node["threshold"]:
            return predict_one(node["left"], value)

        return predict_one(node["right"], value)

    tree = build_tree(list(x), list(y), 0)

    def predict(year):
        return predict_one(tree, year)

    return predict, tree


# ==========================================
# 11. 세 가지 알고리즘 학습
# ==========================================

# 선형회귀 학습
linear_predict, linear_slope, linear_intercept = fit_linear(
    X, Y
)

# 다항회귀 학습
poly_predict, poly_c0, poly_c1, poly_c2 = fit_polynomial(
    Z, Y
)

# 의사결정나무 학습
tree_predict, tree_model = fit_decision_tree(
    X, Y,
    max_depth=3,
    min_leaf=2
)


# ==========================================
# 12. 세 가지 알고리즘 예측 결과
# ==========================================

models = {
    "선형회귀": linear_predict,
    "다항회귀": lambda year: poly_predict(
        (year - center_year) / scale_year
    ),
    "의사결정나무 회귀": tree_predict
}

results = {}

for name, model in models.items():

    predicted = model(prediction_year)

    fitted = [
        model(year) for year in X
    ]

    r2 = calculate_r2(Y, fitted)

    results[name] = {
        "예측기온": predicted,
        "결정계수": r2,
        "학습예측값": fitted
    }


# ==========================================
# 13. 선택 알고리즘 결과 출력
# ==========================================
st.divider()
st.subheader(f"📊 {selected_month}월 예측 결과")

selected_result = results[algorithm]

predicted_temp = selected_result["예측기온"]
r_squared = selected_result["결정계수"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("선택한 알고리즘", algorithm)

with col2:
    st.metric(
        "과거 평균기온",
        f"{Y.mean():.2f}℃"
    )

with col3:
    st.metric(
        f"{prediction_year}년 예측 기온",
        f"{predicted_temp:.2f}℃"
    )

st.success(
    f"🌡️ {prediction_year}년 {selected_month}월 "
    f"예측 평균기온은 {predicted_temp:.2f}℃입니다."
)

st.metric(
    "결정계수 R²",
    f"{r_squared:.4f}"
)


# ==========================================
# 14. 알고리즘별 예측값 비교
# ==========================================
st.divider()
st.subheader("🔍 알고리즘별 예측 결과 비교")

comparison = pd.DataFrame([
    {
        "알고리즘": name,
        "예측 평균기온(℃)": round(result["예측기온"], 2),
        "결정계수(R²)": round(result["결정계수"], 4)
    }
    for name, result in results.items()
])

st.write(
    f"**{prediction_year}년 {selected_month}월** "
    "예측 결과를 비교해 보세요."
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

# 알고리즘별 예측 차이
temperatures = [
    results[name]["예측기온"]
    for name in models
]

difference = max(temperatures) - min(temperatures)

st.info(
    f"💡 세 알고리즘의 예측값 차이는 "
    f"최대 {difference:.2f}℃입니다."
)


# ==========================================
# 15. 선택한 알고리즘의 그래프
# ==========================================
st.divider()
st.subheader(f"📈 {algorithm} 예측 그래프")

fig, ax = plt.subplots(figsize=(13, 6))

# 실제 관측값
ax.scatter(
    X,
    Y,
    color="royalblue",
    s=45,
    label="Actual Temperature",
    zorder=4
)

ax.plot(
    X,
    Y,
    color="skyblue",
    linewidth=1.3,
    alpha=0.6,
    label="Actual Temperature Trend"
)

# 그래프용 연도
start_year = int(min(X))
end_year = prediction_year

line_years = list(range(start_year, end_year + 1))

selected_model = models[algorithm]

line_temps = [
    selected_model(year) for year in line_years
]

# 알고리즘에 따라 그래프 모양 변경
if algorithm == "의사결정나무 회귀":

    ax.step(
        line_years,
        line_temps,
        where="post",
        color="red",
        linewidth=2.5,
        label="Decision Tree Regression",
        zorder=2
    )

else:

    ax.plot(
        line_years,
        line_temps,
        color="red",
        linewidth=2.5,
        linestyle="--",
        label=f"{algorithm} Prediction Line",
        zorder=2
    )

# 실제값과 모델 예측값의 차이
fitted_values = selected_result["학습예측값"]

ax.vlines(
    X,
    fitted_values,
    Y,
    color="gray",
    alpha=0.35,
    linewidth=0.8,
    label="Residuals"
)

# 미래 예측 지점
ax.scatter(
    prediction_year,
    predicted_temp,
    color="red",
    marker="*",
    s=350,
    edgecolors="black",
    label="Predicted Temperature",
    zorder=5
)

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

ax.set_title(
    f"{selected_month}월 평균기온 - {algorithm}",
    fontsize=15
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
# 16. 알고리즘 설명
# ==========================================
st.divider()
st.subheader("🧠 회귀 알고리즘 학습하기")

st.write(
    "회귀 알고리즘은 과거 데이터를 학습하여 "
    "연속적인 숫자 형태의 값을 예측하는 방법입니다."
)

# 선형회귀 설명
if algorithm == "선형회귀":

    st.markdown("### 📏 선형회귀 (Linear Regression)")

    st.latex(r"y = ax + b")

    st.write("""
    선형회귀는 실제 데이터와 가장 가까운 직선을 찾아
    미래의 값을 예측하는 알고리즘입니다.

    - x: 연도
    - y: 평균기온
    - a: 기울기
    - b: y절편

    실제 관측값과 직선의 차이를 제곱하여 합한 값이
    가장 작아지도록 직선을 결정합니다.
    """)

    st.latex(
        rf"y = {linear_slope:.4f}x + {linear_intercept:.2f}"
    )

    st.markdown("**그래프에서 확인할 점**")

    st.write("""
    빨간색 점선이 직선으로 나타납니다.
    실제 기온 데이터가 직선 주변에 얼마나 모여 있는지 확인해 보세요.
    """)


# 다항회귀 설명
elif algorithm == "다항회귀":

    st.markdown("### 📈 다항회귀 (Polynomial Regression)")

    st.latex(
        r"y = b_0 + b_1x + b_2x^2"
    )

    st.write("""
    다항회귀는 선형회귀를 확장한 방법입니다.

    연도와 평균기온의 관계를 2차식으로 표현하여
    직선이 아닌 곡선 형태의 추세를 학습합니다.

    - b₀: 상수항
    - b₁: 1차 계수
    - b₂: 2차 계수

    이번 프로그램에서는 연도를 중심화하고 정규화하여
    계산의 안정성을 높였습니다.
    """)

    st.latex(
        rf"y = {poly_c0:.3f} "
        rf"+ ({poly_c1:.3f})z "
        rf"+ ({poly_c2:.3f})z^2"
    )

    st.write(
        f"여기서 z = (연도 - {center_year:.1f}) / "
        f"{scale_year:.1f} 입니다."
    )

    st.markdown("**그래프에서 확인할 점**")

    st.write("""
    빨간색 곡선이 위로 휘거나 아래로 휘는 모습을 볼 수 있습니다.
    데이터의 곡선 형태 변화가 예측에 반영됩니다.
    """)


# 의사결정나무 설명
else:

    st.markdown(
        "### 🌳 의사결정나무 회귀 (Decision Tree Regression)"
    )

    st.write("""
    의사결정나무 회귀는 데이터를 여러 구간으로 나누고
    각 구간에 속한 데이터의 평균기온을 이용해 예측합니다.

    이번 프로그램은 다음과 같이 작동합니다.

    1. 연도별 기온 데이터를 확인합니다.
    2. 데이터를 여러 연도 구간으로 나눕니다.
    3. 각 구간의 평균기온을 계산합니다.
    4. 새로운 연도가 입력되면 해당 구간의 평균기온을 예측합니다.
    """)

    st.markdown("**이번 프로그램의 설정**")

    st.write("""
    - 최대 깊이: 3
    - 최소 구간 데이터: 2개
    - 분할 기준: 잔차 제곱합이 가장 작아지는 지점
    """)

    st.markdown("**그래프에서 확인할 점**")

    st.write("""
    빨간색 선이 계단 형태로 나타납니다.
    각 계단은 하나의 데이터 구간을 의미합니다.

    미래 연도가 마지막 구간에 속하면 그 구간의 평균기온으로
    예측하므로 미래 구간에서 선이 수평으로 이어질 수 있습니다.
    """)


# ==========================================
# 17. 세 알고리즘 특징 비교
# ==========================================
st.divider()
st.subheader("📚 세 가지 알고리즘 비교하기")

algorithm_info = pd.DataFrame([
    {
        "알고리즘": "선형회귀",
        "그래프 모양": "직선",
        "특징": "전체적인 직선 추세 분석",
        "장점": "단순하고 해석하기 쉬움",
        "한계": "곡선 형태의 변화 표현이 어려움"
    },
    {
        "알고리즘": "다항회귀",
        "그래프 모양": "곡선",
        "특징": "2차 곡선 형태의 추세 분석",
        "장점": "비선형적인 변화 표현 가능",
        "한계": "먼 미래 예측에서 값이 크게 변할 수 있음"
    },
    {
        "알고리즘": "의사결정나무 회귀",
        "그래프 모양": "계단",
        "특징": "데이터를 여러 구간으로 분리",
        "장점": "구간별 변화와 비선형 관계 표현",
        "한계": "학습 범위를 벗어난 미래 예측에 약함"
    }
])

st.dataframe(
    algorithm_info,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 18. 주의사항
# ==========================================
st.divider()

st.warning("""
⚠️ 예측 결과를 해석할 때 주의하세요!

1. 세 알고리즘은 서로 다른 방식으로 데이터를 학습하기 때문에
   예측 결과가 달라질 수 있습니다.

2. 결정계수 R²는 과거 데이터에 대한 설명력을 나타내며,
   미래 예측의 정확도를 직접 보장하지 않습니다.

3. 의사결정나무는 학습 데이터 범위를 벗어난 미래 연도에서
   마지막 구간의 평균값을 예측할 수 있습니다.

4. 실제 기온은 기후변화, 자연현상 등 다양한 요인의 영향을 받습니다.
   따라서 이 프로그램의 결과는 통계적 학습용 예측값입니다.
""")

st.caption(
    "🌍 기온 데이터 분석 | 선형회귀 · 다항회귀 · 의사결정나무 회귀"
)
