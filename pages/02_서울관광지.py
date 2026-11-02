
import streamlit as st
import pandas as pd
import pydeck as pdk

# ==================================
# 1. 페이지 설정
# ==================================

st.set_page_config(
    page_title="서울 관광지 TOP 10",
    page_icon="🗺️",
    layout="wide"
)

# ==================================
# 2. 서울 관광지 데이터
# ==================================

spots = [
    {
        "no": "1",
        "name": "명동",
        "english": "Myeong-dong",
        "lat": 37.5636,
        "lon": 126.9850,
        "category": "쇼핑 · 먹거리",
        "description": "K-뷰티 매장과 패션 숍, 길거리 음식이 모여 있어 쇼핑과 한국 간식을 즐기기 좋아요."
    },
    {
        "no": "2",
        "name": "N서울타워",
        "english": "N Seoul Tower",
        "lat": 37.5512,
        "lon": 126.9882,
        "category": "전망 · 야경",
        "description": "남산 정상에서 서울 도심을 한눈에 볼 수 있는 대표 전망 명소예요."
    },
    {
        "no": "3",
        "name": "코엑스 · 별마당도서관",
        "english": "COEX & Starfield Library",
        "lat": 37.5116,
        "lon": 127.0592,
        "category": "쇼핑 · 문화",
        "description": "대형 쇼핑몰과 별마당도서관이 있어 실내에서 여유롭게 둘러보기 좋아요."
    },
    {
        "no": "4",
        "name": "강남역 일대",
        "english": "Gangnam Station Area",
        "lat": 37.4979,
        "lon": 127.0276,
        "category": "쇼핑 · 트렌드",
        "description": "패션, 카페, 맛집이 밀집한 활기찬 거리로 서울의 최신 유행을 느낄 수 있어요."
    },
    {
        "no": "5",
        "name": "홍대 거리",
        "english": "Hongdae",
        "lat": 37.5563,
        "lon": 126.9236,
        "category": "문화 · 공연",
        "description": "개성 있는 상점과 카페, 거리 공연 등 젊고 자유로운 분위기가 매력적이에요."
    },
    {
        "no": "6",
        "name": "경복궁",
        "english": "Gyeongbokgung Palace",
        "lat": 37.5796,
        "lon": 126.9770,
        "category": "역사 · 전통",
        "description": "조선 시대의 대표 궁궐로 아름다운 건축물과 전통문화를 경험할 수 있어요."
    },
    {
        "no": "7",
        "name": "북촌한옥마을",
        "english": "Bukchon Hanok Village",
        "lat": 37.5826,
        "lon": 126.9830,
        "category": "전통 · 산책",
        "description": "한옥이 이어진 골목에서 한국의 전통적인 주거 풍경을 만날 수 있어요. 주민 생활 공간이므로 조용히 관람해 주세요."
    },
    {
        "no": "8",
        "name": "동대문디자인플라자",
        "english": "Dongdaemun Design Plaza",
        "lat": 37.5665,
        "lon": 127.0092,
        "category": "건축 · 전시",
        "description": "독특한 곡선형 건축물로 유명하며 전시와 디자인 행사가 열리는 복합문화공간이에요."
    },
    {
        "no": "9",
        "name": "광장시장",
        "english": "Gwangjang Market",
        "lat": 37.5700,
        "lon": 126.9997,
        "category": "시장 · 먹거리",
        "description": "빈대떡, 김밥 등 다양한 한국 음식을 맛보고 전통시장의 활기를 느낄 수 있어요."
    },
    {
        "no": "10",
        "name": "남대문시장",
        "english": "Namdaemun Market",
        "lat": 37.5592,
        "lon": 126.9777,
        "category": "시장 · 쇼핑",
        "description": "오랜 역사를 지닌 전통시장으로 기념품과 생활용품, 먹거리를 구경하기 좋아요."
    }
]

df = pd.DataFrame(spots)

# ==================================
# 3. 앱 제목
# ==================================

st.title("🗺️ 외국인 관광객이 즐겨 찾는 서울 명소 TOP 10")

st.markdown("""
서울의 대표 관광지를 지도에서 확인해 보세요! 🇰🇷

지도에 표시된 숫자와 아래 관광지 설명을 함께 확인해 보세요.
""")

st.info(
    "서울의 대표 관광지를 바탕으로 구성한 추천 목록입니다. "
    "공식적인 단일 외국인 선호도 순위는 아닙니다."
)

# ==================================
# 4. 지도 마커 설정
# ==================================

marker_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[lon, lat]",
    get_radius=190,
    radius_min_pixels=8,
    radius_max_pixels=16,
    get_fill_color=[225, 76, 66, 220],
    get_line_color=[255, 255, 255, 255],
    line_width_min_pixels=2,
    pickable=True
)

# ==================================
# 5. 지도 숫자 표시
# ==================================

number_layer = pdk.Layer(
    "TextLayer",
    data=df,
    get_position="[lon, lat]",
    get_text="no",
    get_size=16,
    get_color=[255, 255, 255, 255],
    get_angle=0,
    get_text_anchor="'middle'",
    get_alignment_baseline="'center'",
    size_min_pixels=11,
    size_max_pixels=18,
    pickable=True
)

# ==================================
# 6. 마커 설명 설정
# ==================================

tooltip = {
    "html": """
        <b>{no}. {name}</b><br/>
        {english}<br/>
        {category}
    """,
    "style": {
        "backgroundColor": "#202938",
        "color": "white",
        "padding": "10px"
    }
}

# ==================================
# 7. 서울 지도 생성
# ==================================

deck = pdk.Deck(
    layers=[
        marker_layer,
        number_layer
    ],
    initial_view_state=pdk.ViewState(
        latitude=37.5565,
        longitude=126.9910,
        zoom=11.2,
        pitch=0,
        bearing=0
    ),
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip=tooltip
)

# 지도 출력
st.pydeck_chart(
    deck,
    use_container_width=True,
    height=620
)

# ==================================
# 8. 관광지 설명
# ==================================

st.divider()

st.header("📍 서울 관광지 TOP 10")

st.write(
    "서울의 대표 관광지 10곳을 하나씩 살펴볼까요? 😊"
)

# 관광지 2개씩 나란히 표시
for start in range(0, len(spots), 2):

    cols = st.columns(2)

    for col, spot in zip(cols, spots[start:start + 2]):

        with col:

            with st.container(border=True):

                st.subheader(
                    f"{spot['no']}. {spot['name']}"
                )

                st.caption(
                    f"{spot['english']} | {spot['category']}"
                )

                st.write(
                    spot["description"]
                )

                map_url = (
                    "https://www.google.com/maps/search/?api=1&query="
                    f"{spot['lat']},{spot['lon']}"
                )

                st.markdown(
                    f"[📍 지도에서 위치 보기]({map_url})"
                )

# ==================================
# 9. 마무리
# ==================================

st.divider()

st.markdown("""
### 🌏 서울 여행을 즐겨보세요!

서울에는 역사, 문화, 쇼핑, 먹거리 등
다양한 매력을 가진 관광지가 있어요.

여러분이 가장 가보고 싶은 장소는 어디인가요? 🇰🇷
""")

st.caption(
    "서울 관광지 추천 지도 | 지도 배경: CARTO"
)
