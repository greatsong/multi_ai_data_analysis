import streamlit as st
import pandas as pd
import openai

# ——————————— 설정 ———————————
st.set_page_config(page_title="데이터 챗봇 분석기", layout="wide")

# OpenAI API 키 로드
openai.api_key = st.secrets.get("openai_api_key", "")

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
        try:
            if uploaded_file.name.lower().endswith("csv"):
                # 다양한 인코딩 자동 시도
                encodings = [None, 'utf-8-sig', 'euc-kr', 'cp949', 'latin1']
                for enc in encodings:
                    try:
                        uploaded_file.seek(0)
                        if enc:
                            df = pd.read_csv(uploaded_file, encoding=enc)
                        else:
                            df = pd.read_csv(uploaded_file)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise UnicodeDecodeError(
                        "utf-8", b"", 0, 1, "Unable to decode with tried encodings"
                    )
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success(f"파일이 업로드되었습니다! (행: {df.shape[0]}, 열: {df.shape[1]})")
            with st.expander("데이터 미리보기"):
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    # 모델 선택 기능 추가
    st.markdown("---")
    model = st.selectbox(
        "🤖 사용할 AI 모델 선택", 
        options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=0
    )

# ——————————— 메인 영역 ———————————
st.title("🧑‍💻 데이터 챗봇 분석기")
st.markdown(
    "업로드된 데이터를 기반으로 질문하세요. 선택된 모델로 GPT가 판다스 코드나 분석 결과를 반환합니다."
)

# 채팅 UI
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("질문 입력…", "")
    submit = st.form_submit_button("전송")

if submit and user_input:
    # 데이터 샘플 제공
    if st.session_state.df is not None:
        sample_csv = st.session_state.df.head(5).to_csv(index=False)
        context = f"업로드된 데이터 첫 5행:\n{sample_csv}\n\n"
    else:
        context = "데이터가 업로드되지 않았습니다.\n\n"

    st.session_state.messages.append({"role": "user", "content": context + user_input})

    with st.spinner("분석 중…"):
        # 새로운 API 방식 사용
        resp = openai.chat.completions.create(
            model=model,
            messages=st.session_state.messages,
            temperature=0.2,
        )
    answer = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})

# 대화 내역 출력
for msg in st.session_state.messages[1:]:
    if msg["role"] == "user":
        st.markdown(f"**나:** {msg['content']}")
    else:
        st.markdown("**GPT:**")
        st.code(msg["content"], language="python")

# ——————————— 푸터 ———————————
st.markdown("---")
st.markdown(
    "※ CSV 파일 디코딩 시 utf-8, utf-8-sig, euc-kr, cp949, latin1 순으로 시도합니다. `temperature=0.0` 또는 `max_tokens` 조정으로 비용을 관리해 보세요."
)
