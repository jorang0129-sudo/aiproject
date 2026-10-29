import streamlit as st
st.title('나의 첫 웹 사이트 만들기!')
a=st.text_input('이름을 입력해주세요')
b=st.selectbox('좋아하는 음식을 선택하세요!',['분모자중국당면콘마요퐁당납작만두를추가한 엽떡','까르보불닭과 뿌링클 콤보','마틸다생초코꾸덕케이크'])
if st.button('인사말 생성'):
  st.write('반갑습니다!')
  st.info(a+'님! 안녕하세요!')
  st.warning(b+'를 좋아하시네요! 맛잘알!')
  st.error('잘 부탁드립니다!')
  st.balloons()
