    selected_district = st.selectbox("행정구 선택", districts)

with col2:
    # 현재 CSV에는 남성/여성 구분 열이 없습니다.
    selected_gender = st.selectbox(
        "성별 선택",
        ["전체 인구 (성별 구분 없음)"],
        help="현재 CSV는 성별을 나누지 않은 전체 인구 자료입니다."
    )

with col3:
    selected_year = st.selectbox(
        "연도 선택",
        years,
        index=len(years) - 1,
        format_func=lambda year: f"{year}년"
    )


# -----------------------------------------
# 선택한 행정구·연도의 연령별 인구 추출
# -----------------------------------------
selected_rows = df[df["행정구역"] == selected_district]

if selected_rows.empty:
    st.error("선택한 행정구의 데이터를 찾을 수 없습니다.")
    st.stop()

row = selected_rows.iloc[0]
values = []

for age in AGE_GROUPS:
    column_name = f"{selected_year}년_거주자_{age}"
    if column_name in df.columns and pd.notna(row[column_name]):
        values.append(int(row[column_name]))
    else:
        values.append(0)

chart_data = pd.DataFrame({
    "나이": AGE_GROUPS,
    "인구수": values
})

total = sum(values)

m1, m2, m3 = st.columns(3)
m1.metric("선택한 행정구", selected_district)
m2.metric("선택 연도", f"{selected_year}년")
m3.metric("연령별 인구 합계", f"{total:,}명")


# -----------------------------------------
# 하늘색 꺾은선 그래프
# -----------------------------------------
st.subheader("📈 연령대별 인구수 그래프")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_data["나이"],
        y=chart_data["인구수"],
        mode="lines+markers",
        name="인구수",
        line=dict(color="#87CEEB", width=3),
        marker=dict(color="#87CEEB", size=8),
        hovertemplate="나이: %{x}<br>인구수: %{y:,.0f}명<extra></extra>"
    )
)

max_value = max(values) if values else 0
y_max = max(100, ((max_value // 100) + 1) * 100)

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
    margin=dict(l=80, r=30, t=80, b=80)
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "가로축은 10세 단위 연령 구간으로 표시하며, "
    "세로축 눈금과 격자 간격은 요청에 따라 100명으로 설정했습니다."
)

with st.expander("📋 인구 데이터 표 보기"):
    table = chart_data.copy()
    table["인구수"] = table["인구수"].map(lambda n: f"{n:,}명")
    st.dataframe(table, use_container_width=True, hide_index=True)

st.info(
    "성별 안내: 현재 population.csv에는 남성과 여성의 인구를 구분하는 "
    "열이 없습니다. 따라서 전체 인구만 표시됩니다. "
    "남녀별 분석을 하려면 성별 구분 데이터가 포함된 CSV가 필요합니다."
)
