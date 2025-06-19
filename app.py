import streamlit as st
import pandas as pd
import openai

# ——————————— 설정 ———————————
st.set_page_config(page_title="데이터 챗봇 분석기", layout="wide")

# OpenAI API 키 로드
openai.api_key = st.secrets["openai_api_key"]

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "당신은 데이터 분석을 도와주는 친절한 어시스턴트입니다."}
    ]
if "df" not in st.session_state:
    st.session_state.df = None

# ——————————— 사이드바 ———————————
with st.sidebar:
    st.title("📂 데이터 업로드 & 설정")
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일 업로드",
        type=["csv", "xls", "xlsx"]
    )
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("파일이 업로드되었습니다! (행:{}, 열:{})".format(*df.shape))
        with st.expander("데이터 미리보기"):
            st.dataframe(df.head())

    # 모델 선택 기능 추가
    st.markdown("---")
    model = st.selectbox(
        "🤖 사용할 AI 모델 선택", 
        options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=0
    )

# ——————————— 메인 영역 ———————————
st.title("🧑‍💻 데이터 챗봇 분석기")
st.markdown("데이터를 업로드한 후, 아래에 질문을 입력하세요. 선택한 모델로 GPT가 판다스 코드나 분석 결과를 반환합니다.")

# 채팅 UI
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("질문 입력…", "")
    submit = st.form_submit_button("전송")

if submit and user_input:
    # DataFrame 샘플 삽입
    if st.session_state.df is not None:
        sample = st.session_state.df.head(5).to_csv(index=False)
        context = f"다음은 업로드된 데이터의 처음 5줄입니다:\n{sample}\n\n"
    else:
        context = "아직 데이터가 업로드되지 않았습니다.\n\n"

    # 메시지 업데이트
    st.session_state.messages.append({"role": "user", "content": context + user_input})

    # OpenAI 호출
    with st.spinner("분석 중…"):
        resp = openai.ChatCompletion.create(
            model=model,
            messages=st.session_state.messages,
            temperature=0.2,
        )
    answer = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})

# 이전 대화 보기
for msg in st.session_state.messages[1:]:  # system 제외
    if msg["role"] == "user":
        st.markdown(f"**나:** {msg['content']}")
    else:
        st.markdown("**GPT:**")
        st.code(msg["content"], language="python")

# ——————————— 푸터 ———————————
st.markdown("---")
st.markdown("※ API 사용량이 걱정되면, `temperature=0.0` 으로 낮추거나 `max_tokens` 를 조정해 보세요.")
