
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# =====================================
# 1. 페이지 기본 설정
# =====================================

st.set_page_config(
    page_title="붓꽃 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 인공지능 붓꽃 품종 예측")
st.write("꽃의 크기를 입력하면 인공지능이 붓꽃의 품종을 예측해 줘요!")

st.divider()

# =====================================
# 2. CSV 파일 불러오기
# =====================================

# 현재 py 파일은 pages 폴더에 있고
# CSV 파일은 상위 폴더에 있음

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
# 3. 사용할 데이터 설정
# =====================================

features = [
    "꽃받침 길이",
    "꽃잎 길이",
    "꽃잎 너비"
]

target = "품종"

# 학습 데이터
X = df[features]
y = df[target]

# =====================================
# 4. 랜덤 포레스트 모델 생성 및 학습
# =====================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# =====================================
# 5. 입력값 범위 안내
# =====================================

st.subheader("📏 꽃의 크기 입력하기")

st.info(
    "아래 범위는 업로드한 Iris.csv 데이터에 "
    "실제로 기록된 최솟값과 최댓값이에요."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("꽃받침 길이", "4.3 ~ 7.9 cm")

with col2:
    st.metric("꽃잎 길이", "1.0 ~ 6.9 cm")

with col3:
    st.metric("꽃잎 너비", "0.1 ~ 2.5 cm")

st.caption("※ 입력값은 Iris.csv의 실제 측정 범위 안에서 선택할 수 있어요.")

# =====================================
# 6. 사용자 입력
# =====================================

st.subheader("✏️ 측정값 입력")

col1, col2, col3 = st.columns(3)

with col1:
    sepal_length = st.slider(
        "꽃받침 길이 (cm)",
        min_value=4.3,
        max_value=7.9,
        value=5.8,
        step=0.1
    )

with col2:
    petal_length = st.slider(
        "꽃잎 길이 (cm)",
        min_value=1.0,
        max_value=6.9,
        value=3.8,
        step=0.1
    )

with col3:
    petal_width = st.slider(
        "꽃잎 너비 (cm)",
        min_value=0.1,
        max_value=2.5,
        value=1.2,
        step=0.1
    )

# 입력값 데이터프레임 생성
input_data = pd.DataFrame(
    [[sepal_length, petal_length, petal_width]],
    columns=features
)

# =====================================
# 7. 품종 예측
# =====================================

st.divider()

st.subheader("🤖 인공지능 예측 결과")

prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

if prediction == "Iris-setosa":
    flower_name = "세토사 (Setosa)"
    flower_emoji = "🌷"

elif prediction == "Iris-versicolor":
    flower_name = "버시컬러 (Versicolor)"
    flower_emoji = "🌼"

else:
    flower_name = "버지니카 (Virginica)"
    flower_emoji = "🌺"

st.success(f"예측된 붓꽃 품종: {flower_emoji} {flower_name}")

# 예측 확률 표시
st.write("### 📊 품종별 예측 확률")

prob_col1, prob_col2, prob_col3 = st.columns(3)

for col, species, probability in zip(
    [prob_col1, prob_col2, prob_col3],
    model.classes_,
    probabilities
):
    with col:
        species_name = species.replace("Iris-", "")
        st.metric(
            label=species_name,
            value=f"{probability * 100:.1f}%"
        )
        st.progress(float(probability))

# =====================================
# 8. 입력값 그래프로 표현
# =====================================

st.divider()

st.subheader("📈 꽃잎 크기 분포 그래프")

st.write(
    "가로축은 꽃잎 길이, 세로축은 꽃잎 너비입니다. "
    "초록색 별(★)은 사용자가 입력한 값이에요."
)

fig, ax = plt.subplots(figsize=(9, 6))

# 품종별 데이터 표시
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

# 사용자가 입력한 값: 초록색 별
ax.scatter(
    petal_length,
    petal_width,
    color="green",
    marker="*",
    s=350,
    edgecolors="black",
    linewidths=1,
    label="My Input",
    zorder=10
)

# 그래프 제목과 축 설정
ax.set_title(
    "Iris Petal Length vs Petal Width",
    fontsize=15
)

ax.set_xlabel("Petal Length (cm)", fontsize=12)
ax.set_ylabel("Petal Width (cm)", fontsize=12)

ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# =====================================
# 9. 입력값 확인
# =====================================

st.subheader("🔍 내가 입력한 측정값")

st.dataframe(
    input_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "이 예측은 업로드한 Iris.csv 전체 데이터를 "
    "학습한 랜덤 포레스트 모델을 이용한 결과입니다."
)

st.divider()

st.write("🌸 꽃의 크기를 바꿔가며 품종 예측 결과를 확인해 보세요!")
