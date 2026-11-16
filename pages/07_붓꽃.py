
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="🌸 봄날의 붓꽃 예측",
    page_icon="🌸",
    layout="wide"
)


# --------------------------------------------------
# 핑크 봄 테마 CSS
# --------------------------------------------------
st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background: linear-gradient(
        180deg,
        #fff0f6 0%,
        #fff7fa 45%,
        #ffffff 100%
    );
}

/* 기본 글자 */
html, body, [class*="css"] {
    color: #75445d;
}

/* 메인 제목 */
h1 {
    color: #e75480 !important;
    text-align: center;
    font-weight: 900 !important;
    text-shadow: 2px 2px 5px #ffc5d8;
}

/* 소제목 */
h2, h3 {
    color: #d94f83 !important;
}

/* 설명 문구 */
p {
    color: #80556a;
}

/* 버튼 */
.stButton > button {
    background: linear-gradient(90deg, #ff8eb5, #f56f9d);
    color: white;
    border: none;
    border-radius: 20px;
    font-weight: bold;
}

/* 선택 상자 */
div[data-baseweb="select"] > div {
    background-color: #fff8fb;
    border: 1px solid #ffc1d5;
    border-radius: 12px;
}

/* 입력 슬라이더 */
.stSlider [data-baseweb="slider"] div {
    background-color: #f58db0;
}

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: #ffe5ef;
    border-radius: 15px;
    padding: 8px;
}

.stTabs [data-baseweb="tab"] {
    color: #c04d79;
    font-weight: bold;
    border-radius: 12px;
}

.stTabs [aria-selected="true"] {
    background-color: #f58db0 !important;
    color: white !important;
}

/* 데이터프레임 */
.stDataFrame {
    background-color: #fff8fb;
    border-radius: 15px;
}

/* 안내 상자 */
div[data-testid="stAlert"] {
    background-color: #fff0f6;
    border-radius: 15px;
}

/* 꽃잎 애니메이션 */
.flower {
    position: fixed;
    top: -10%;
    z-index: 9999;
    pointer-events: none;
    animation-name: fall;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    opacity: 0.8;
}

@keyframes fall {
    0% {
        transform: translateY(-10vh) rotate(0deg);
    }
    100% {
        transform: translateY(115vh) rotate(360deg);
    }
}

/* 꽃잎 위치와 속도 */
.flower:nth-child(1) {
    left: 5%;
    animation-duration: 12s;
    font-size: 22px;
}

.flower:nth-child(2) {
    left: 15%;
    animation-duration: 17s;
    animation-delay: 2s;
    font-size: 28px;
}

.flower:nth-child(3) {
    left: 25%;
    animation-duration: 14s;
    animation-delay: 5s;
    font-size: 20px;
}

.flower:nth-child(4) {
    left: 38%;
    animation-duration: 19s;
    animation-delay: 1s;
    font-size: 25px;
}

.flower:nth-child(5) {
    left: 50%;
    animation-duration: 13s;
    animation-delay: 4s;
    font-size: 30px;
}

.flower:nth-child(6) {
    left: 62%;
    animation-duration: 16s;
    animation-delay: 3s;
    font-size: 23px;
}

.flower:nth-child(7) {
    left: 73%;
    animation-duration: 18s;
    animation-delay: 6s;
    font-size: 27px;
}

.flower:nth-child(8) {
    left: 85%;
    animation-duration: 15s;
    animation-delay: 2s;
    font-size: 21px;
}

.flower:nth-child(9) {
    left: 95%;
    animation-duration: 20s;
    animation-delay: 7s;
    font-size: 26px;
}

/* 카드 디자인 */
.result-card {
    background: linear-gradient(135deg, #fff0f6, #ffe0ec);
    border: 2px solid #ffc1d5;
    border-radius: 20px;
    padding: 20px;
    margin: 8px 0;
    text-align: center;
    box-shadow: 0 4px 12px #f7c4d5;
}

.result-card h3 {
    color: #d94f83;
}

.result-card h2 {
    color: #e75480;
}

/* 구분선 */
hr {
    border-color: #ffc1d5;
}

</style>

<div class="flower">🌸</div>
<div class="flower">🌷</div>
<div class="flower">🌸</div>
<div class="flower">🌺</div>
<div class="flower">🌸</div>
<div class="flower">🌷</div>
<div class="flower">🌸</div>
<div class="flower">🌺</div>
<div class="flower">🌸</div>

""", unsafe_allow_html=True)


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🌸 봄날의 붓꽃 품종 예측 🌸")

st.markdown(
    """
    <div style="
        text-align:center;
        background:linear-gradient(90deg,#ffe1ec,#fff5f8,#ffe1ec);
        padding:18px;
        border-radius:20px;
        margin-bottom:20px;
        border:2px solid #ffc1d5;
    ">
        <h3 style="color:#d94f83;">
            🌷 꽃의 길이와 너비로 붓꽃의 품종을 알아봐요! 🌷
        </h3>
        <p>
            세 가지 인공지능 알고리즘이 어떤 품종을 예측하는지 비교해 보세요.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# CSV 파일 불러오기
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "Iris.csv"

try:
    try:
        df = pd.read_csv(CSV_PATH, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

except FileNotFoundError:
    st.error("Iris.csv 파일을 찾을 수 없어요. 파일 위치를 확인해 주세요.")
    st.stop()


# --------------------------------------------------
# 데이터 확인
# --------------------------------------------------
required_columns = [
    "꽃받침 길이",
    "꽃받침 너비",
    "꽃잎 길이",
    "꽃잎 너비",
    "품종"
]

if not all(col in df.columns for col in required_columns):
    st.error("CSV 파일의 열 이름을 확인해 주세요.")
    st.write("현재 CSV 열 이름:", list(df.columns))
    st.stop()

df = df.dropna(subset=required_columns)

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()

X = df[required_columns[:4]]
y = df["품종"]


# --------------------------------------------------
# 모델 만들기
# --------------------------------------------------
def create_models():

    models = {
        "의사결정나무": DecisionTreeClassifier(
            random_state=42
        ),

        "랜덤포레스트": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    return models


# --------------------------------------------------
# 예측 및 비교 함수
# --------------------------------------------------
def run_prediction(features, x_label, y_label, tab_key):

    st.subheader("🌷 꽃의 특징 입력하기")

    col1, col2 = st.columns(2)

    with col1:
        x_min = float(df[x_label].min())
        x_max = float(df[x_label].max())

        x_value = st.slider(
            x_label + " (cm)",
            min_value=x_min,
            max_value=x_max,
            value=float(df[x_label].median()),
            step=0.1,
            key=tab_key + "_x"
        )

    with col2:
        y_min = float(df[y_label].min())
        y_max = float(df[y_label].max())

        y_value = st.slider(
            y_label + " (cm)",
            min_value=y_min,
            max_value=y_max,
            value=float(df[y_label].median()),
            step=0.1,
            key=tab_key + "_y"
        )

    input_data = pd.DataFrame(
        [[x_value, y_value]],
        columns=features
    )

    st.markdown("---")
    st.subheader("🌸 알고리즘별 예측 결과")

    models = create_models()

    results = []
    probability_results = []

    for model_name, model in models.items():

        model.fit(df[features], y)

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        class_names = model.classes_

        predicted_probability = max(probabilities) * 100

        results.append({
            "알고리즘": model_name,
            "예측 품종": prediction,
            "예측 확률": f"{predicted_probability:.1f}%"
        })

        probability_row = {"알고리즘": model_name}

        for class_name, probability in zip(
            class_names, probabilities
        ):
            probability_row[class_name] = round(
                probability * 100, 1
            )

        probability_results.append(probability_row)

    # 알고리즘 비교표
    result_df = pd.DataFrame(results)

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    # 알고리즘별 결과 카드
    card_cols = st.columns(3)

    for i, result in enumerate(results):

        with card_cols[i]:
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>🌸 {result['알고리즘']}</h3>
                    <h2>{result['예측 품종']}</h2>
                    <p>예측 확률</p>
                    <h2>{result['예측 확률']}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 품종별 확률 비교
    st.markdown("---")
    st.subheader("🌷 품종별 예측 확률 비교")

    probability_df = pd.DataFrame(probability_results)

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

    # 그래프
    st.markdown("---")
    st.subheader("🌸 붓꽃 데이터 시각화")

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ["#f4a7c5", "#c5a3e8", "#f7c98b"]

    varieties = list(df["품종"].unique())

    for i, variety in enumerate(varieties):

        variety_data = df[df["품종"] == variety]

        ax.scatter(
            variety_data[x_label],
            variety_data[y_label],
            label=variety,
            color=colors[i % len(colors)],
            alpha=0.75,
            s=55
        )

    # 사용자 입력 위치
    ax.scatter(
        x_value,
        y_value,
        color="#e60073",
        marker="*",
        s=350,
        edgecolors="white",
        linewidths=1.5,
        label="내가 입력한 꽃",
        zorder=5
    )

    ax.set_xlabel(x_label + " (cm)")
    ax.set_ylabel(y_label + " (cm)")
    ax.set_title("🌸 붓꽃의 특징 분포")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.patch.set_facecolor("#fff7fa")
    ax.set_facecolor("#fffafd")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 탭 구성
# --------------------------------------------------
tab1, tab2 = st.tabs([
    "🌸 꽃잎 속성으로 예측하기",
    "🌷 꽃받침 속성으로 예측하기"
])


# --------------------------------------------------
# 탭 1: 꽃잎 속성
# --------------------------------------------------
with tab1:

    st.markdown("### 🌸 꽃잎의 길이와 너비로 품종 예측")

    run_prediction(
        features=["꽃잎 길이", "꽃잎 너비"],
        x_label="꽃잎 길이",
        y_label="꽃잎 너비",
        tab_key="petal"
    )


# --------------------------------------------------
# 탭 2: 꽃받침 속성
# --------------------------------------------------
with tab2:

    st.markdown("### 🌷 꽃받침의 길이와 너비로 품종 예측")

    run_prediction(
        features=["꽃받침 길이", "꽃받침 너비"],
        x_label="꽃받침 길이",
        y_label="꽃받침 너비",
        tab_key="sepal"
    )


# --------------------------------------------------
# 알고리즘 설명
# --------------------------------------------------
st.markdown("---")
st.header("🌼 세 가지 알고리즘 알아보기")

with st.expander("🌸 의사결정나무 (Decision Tree)"):

    st.write("""
    🌱 데이터를 여러 기준으로 나누면서 품종을 예측하는 방법이에요.

    - 질문을 통해 가지를 나누듯 데이터를 분류해요.
    - 구조가 단순해서 결과를 이해하기 쉬워요.
    """)

with st.expander("🌸 랜덤포레스트 (Random Forest)"):

    st.write("""
    🌱 여러 개의 의사결정나무를 만들고 결과를 모아서 예측하는 방법이에요.

    - 여러 나무의 예측 결과를 종합해요.
    - 하나의 나무보다 안정적인 예측을 기대할 수 있어요.
    """)

with st.expander("🌸 KNN (K-Nearest Neighbors)"):

    st.write("""
    🌱 새로 입력한 꽃과 가장 가까운 데이터들의 품종을 참고하는 방법이에요.

    - 가까운 이웃 5개의 데이터를 참고해요.
    - 주변에 어떤 품종이 많은지 확인해 예측해요.
    - 거리 계산을 위해 데이터를 표준화해요.
    """)


# --------------------------------------------------
# 마지막 안내
# --------------------------------------------------
st.markdown("""
<div style="
    background:linear-gradient(90deg,#ffe1ec,#fff5f8,#ffe1ec);
    border-radius:20px;
    padding:20px;
    text-align:center;
    border:2px solid #ffc1d5;
">
    <h3>🌸 오늘도 예쁜 꽃처럼 행복한 하루! 🌸</h3>
    <p>붓꽃 예측 프로그램을 이용해 줘서 고마워요 💗</p>
    <p>🌷 🌸 🌷 🌸 🌷 🌸 🌷</p>
</div>
""", unsafe_allow_html=True)
