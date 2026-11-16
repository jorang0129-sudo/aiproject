
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="붓꽃 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 인공지능 붓꽃 품종 예측")
st.write(
    "꽃의 크기를 입력하면 인공지능이 "
    "붓꽃의 품종을 예측해 줘요!"
)

st.divider()

# =====================================
# 2. CSV 파일 불러오기
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "Iris.csv"

try:
    try:
        df = pd.read_csv(CSV_PATH, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

except FileNotFoundError:
    st.error("Iris.csv 파일을 찾을 수 없습니다.")
    st.stop()

# =====================================
# 3. 데이터 설정
# =====================================

features = [
    "꽃받침 길이",
    "꽃잎 길이",
    "꽃잎 너비"
]

target = "품종"

# 열 이름 확인
if not all(col in df.columns for col in features + [target]):
    st.error(
        "CSV 파일의 열 이름이 코드와 일치하지 않습니다. "
        "Iris.csv의 열 이름을 확인해 주세요."
    )
    st.stop()

# 결측치 확인
if df[features + [target]].isnull().any().any():
    st.error("데이터에 결측치가 있습니다. CSV 파일을 확인해 주세요.")
    st.stop()

X = df[features]
y = df[target]

# =====================================
# 4. 분류 알고리즘 3개 생성
# =====================================

models = {

    "의사결정나무 (Decision Tree)": DecisionTreeClassifier(
        random_state=42
    ),

    "랜덤 포레스트 (Random Forest)": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "K-최근접 이웃 (KNN)": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

}

# =====================================
# 5. 입력값 범위 안내
# =====================================

st.subheader("📏 꽃의 크기 입력 범위")

st.info(
    "아래 범위는 Iris.csv에 기록된 "
    "실제 최솟값과 최댓값입니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "꽃받침 길이",
        f"{df['꽃받침 길이'].min():.1f} ~ "
        f"{df['꽃받침 길이'].max():.1f} cm"
    )

with col2:
    st.metric(
        "꽃잎 길이",
        f"{df['꽃잎 길이'].min():.1f} ~ "
        f"{df['꽃잎 길이'].max():.1f} cm"
    )

with col3:
    st.metric(
        "꽃잎 너비",
        f"{df['꽃잎 너비'].min():.1f} ~ "
        f"{df['꽃잎 너비'].max():.1f} cm"
    )

st.caption(
    "※ 입력값은 데이터에 기록된 범위 안에서 선택할 수 있습니다."
)

# =====================================
# 6. 사용자 입력
# =====================================

st.divider()

st.subheader("✏️ 꽃의 크기 입력하기")

col1, col2, col3 = st.columns(3)

with col1:
    sepal_length = st.slider(
        "꽃받침 길이 (cm)",
        min_value=float(df["꽃받침 길이"].min()),
        max_value=float(df["꽃받침 길이"].max()),
        value=5.8,
        step=0.1
    )

with col2:
    petal_length = st.slider(
        "꽃잎 길이 (cm)",
        min_value=float(df["꽃잎 길이"].min()),
        max_value=float(df["꽃잎 길이"].max()),
        value=3.8,
        step=0.1
    )

with col3:
    petal_width = st.slider(
        "꽃잎 너비 (cm)",
        min_value=float(df["꽃잎 너비"].min()),
        max_value=float(df["꽃잎 너비"].max()),
        value=1.2,
        step=0.1
    )

# 입력 데이터 생성
input_data = pd.DataFrame(
    [[sepal_length, petal_length, petal_width]],
    columns=features
)

# =====================================
# 7. 알고리즘 선택
# =====================================

st.divider()

st.subheader("🤖 분류 알고리즘 선택")

selected_model = st.selectbox(
    "사용할 알고리즘을 선택하세요.",
    list(models.keys())
)

# =====================================
# 8. 모델 학습 및 예측
# =====================================

model = models[selected_model]

model.fit(X, y)

prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

# =====================================
# 9. 예측 결과 출력
# =====================================

st.divider()

st.subheader("🔮 붓꽃 품종 예측 결과")

flower_names = {
    "Iris-setosa": "세토사 (Setosa) 🌷",
    "Iris-versicolor": "버시컬러 (Versicolor) 🌼",
    "Iris-virginica": "버지니카 (Virginica) 🌺"
}

st.success(
    f"예측된 품종은 {flower_names[prediction]} 입니다!"
)

st.write("### 📊 품종별 예측 확률")

prob_cols = st.columns(3)

for col, species, probability in zip(
    prob_cols,
    model.classes_,
    probabilities
):

    with col:

        name = species.replace("Iris-", "")

        st.metric(
            label=name,
            value=f"{probability * 100:.1f}%"
        )

        st.progress(float(probability))

# =====================================
# 10. 꽃잎 크기 그래프
# =====================================

st.divider()

st.subheader("📈 꽃잎 길이와 너비 그래프")

st.write(
    "가로축은 꽃잎 길이, 세로축은 꽃잎 너비입니다. "
    "초록색 별(★)은 사용자가 입력한 위치입니다."
)

fig, ax = plt.subplots(figsize=(9, 6))

colors = {
    "Iris-setosa": "blue",
    "Iris-versicolor": "orange",
    "Iris-virginica": "red"
}

labels = {
    "Iris-setosa": "Setosa",
    "Iris-versicolor": "Versicolor",
    "Iris-virginica": "Virginica"
}

# 기존 데이터 표시
for species in df[target].unique():

    species_data = df[df[target] == species]

    ax.scatter(
        species_data["꽃잎 길이"],
        species_data["꽃잎 너비"],
        color=colors[species],
        label=labels[species],
        alpha=0.65,
        s=55
    )

# 사용자 입력값 초록색 별 표시
ax.scatter(
    petal_length,
    petal_width,
    color="green",
    marker="*",
    s=400,
    edgecolors="black",
    linewidths=1,
    label="My Input",
    zorder=10
)

ax.set_title(
    "Iris Petal Length vs Petal Width",
    fontsize=15
)

ax.set_xlabel("Petal Length (cm)", fontsize=12)
ax.set_ylabel("Petal Width (cm)", fontsize=12)

ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

plt.close(fig)

# =====================================
# 11. 입력값 확인
# =====================================

st.divider()

st.subheader("🔍 내가 입력한 측정값")

st.dataframe(
    input_data,
    use_container_width=True,
    hide_index=True
)

# =====================================
# 12. 알고리즘 작동 원리 설명
# =====================================

st.divider()

st.title("📚 분류 알고리즘 작동 원리")

st.write(
    "이번 앱에서 사용한 3가지 인공지능 알고리즘을 알아볼까요?"
)

# -------------------------------------
# 알고리즘 1: 의사결정나무
# -------------------------------------

with st.expander(
    "🌳 1. 의사결정나무 (Decision Tree)",
    expanded=True
):

    st.write("""
    **작동 원리**

    의사결정나무는 데이터를 여러 가지 질문으로 나누면서
    품종을 구분하는 알고리즘입니다.

    ① 꽃잎 길이가 얼마나 되는지 확인합니다.

    ② 꽃잎 너비가 얼마나 되는지 확인합니다.

    ③ 질문에 대한 답을 따라가며 최종 품종을 결정합니다.

    **특징**

    - 나무의 가지처럼 질문을 단계별로 나눕니다.
    - 작동 원리를 이해하기 쉽습니다.
    - 데이터에 지나치게 맞춰 학습하면 과적합이 발생할 수 있습니다.
    """)

# -------------------------------------
# 알고리즘 2: 랜덤 포레스트
# -------------------------------------

with st.expander(
    "🌲 2. 랜덤 포레스트 (Random Forest)"
):

    st.write("""
    **작동 원리**

    랜덤 포레스트는 여러 개의 의사결정나무를 만들어
    각각 예측한 결과를 종합하는 알고리즘입니다.

    ① 여러 개의 학습용 데이터를 무작위로 선택합니다.

    ② 각각의 데이터로 여러 의사결정나무를 학습시킵니다.

    ③ 각 나무가 붓꽃의 품종을 예측합니다.

    ④ 가장 많은 나무가 선택한 품종을 최종 결과로 결정합니다.

    **특징**

    - 여러 의사결정나무의 결과를 종합합니다.
    - 단일 의사결정나무보다 예측이 안정적인 경우가 많습니다.
    - 여러 나무를 사용하므로 계산량이 늘어날 수 있습니다.
    """)

# -------------------------------------
# 알고리즘 3: KNN
# -------------------------------------

with st.expander(
    "📍 3. K-최근접 이웃 (KNN)"
):

    st.write("""
    **작동 원리**

    KNN은 새로운 데이터와 가장 가까운 기존 데이터들을 찾아
    주변 데이터의 품종을 참고하는 알고리즘입니다.

    ① 사용자가 입력한 꽃의 크기를 확인합니다.

    ② 기존 데이터에서 가까운 데이터 5개를 찾습니다.

    ③ 가까운 데이터 5개의 품종을 확인합니다.

    ④ 가장 많이 나타난 품종을 예측 결과로 결정합니다.

    **특징**

    - 주변 데이터의 영향을 많이 받습니다.
    - 작동 원리가 직관적입니다.
    - 데이터의 크기와 단위를 표준화하는 것이 중요합니다.

    ※ 이 앱에서는 KNN에 표준화 과정을 적용했습니다.
    """)

# =====================================
# 13. 마무리
# =====================================

st.divider()

st.success(
    "🌸 측정값과 알고리즘을 바꿔가며 "
    "붓꽃 품종 예측 결과를 비교해 보세요!"
)

st.caption(
    "본 프로그램은 Iris.csv 전체 데이터를 학습하여 "
    "사용자가 입력한 측정값의 품종을 예측합니다."
)
