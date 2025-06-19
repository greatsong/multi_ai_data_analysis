import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- 페이지 설정 및 기본 스타일 ---
st.set_page_config(
    page_title="데이터 분석 코드 생성기",
    page_icon="📊",
    layout="wide"
)

# Gemini API 키 설정
# Streamlit Cloud의 Secrets에서 API 키를 가져옵니다.
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    GEMINI_API_AVAILABLE = True
except (KeyError, AttributeError):
    st.error("⚠️ Gemini API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
    GEMINI_API_AVAILABLE = False

# --- 함수 정의 ---

def load_data(uploaded_file):
    """업로드된 파일을 pandas DataFrame으로 변환합니다."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.warning("지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 업로드해주세요.")
            return None
        return df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

def get_gemini_response(user_prompt, df_info):
    """Gemini API에 요청을 보내고 응답을 스트리밍으로 받습니다."""
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    # Gemini에게 전달할 시스템 명령어 (역할 부여)
    system_instruction = f"""
    당신은 데이터 분석 전문가이며, Streamlit 코드를 생성하는 AI 어시스턴트입니다.
    당신의 임무는 사용자의 한글 요청과 제공된 데이터프레임의 정보를 바탕으로, Streamlit에서 바로 실행 가능한 Python 코드를 생성하는 것입니다.

    **규칙:**
    1.  데이터는 이미 `df`라는 이름의 pandas DataFrame으로 로드되어 있다고 가정하세요. (데이터 로딩 코드는 절대 포함하지 마세요.)
    2.  `import streamlit as st`와 `import pandas as pd`는 이미 선언되어 있다고 가정하세요.
    3.  오직 요청된 작업을 수행하는 Streamlit 코드 블록만 생성하세요. 코드 외의 설명이나 주석은 최소화하고, 코드 블록으로 감싸서 제공하세요.
    4.  데이터 시각화 라이브러리가 필요하다면, `plotly.express` 또는 `matplotlib.pyplot` 사용을 권장합니다.
    5.  사용자가 이해하기 쉽고 직관적인 코드를 작성해주세요.
    
    **제공된 데이터프레임 정보:**
    {df_info}
    """
    
    # 모델에 프롬프트 전달
    response = model.generate_content(
        [system_instruction, user_prompt],
        stream=True
    )
    return response

# --- 메인 앱 구성 ---

st.title("📊 AI 데이터 분석 코드 생성 도우미")
st.markdown("---")

# 사이드바 구성
with st.sidebar:
    st.header("1. 데이터 업로드")
    uploaded_file = st.file_uploader("CSV 또는 Excel 파일을 업로드하세요.", type=["csv", "xls", "xlsx"])
    
    if uploaded_file:
        st.success(f"'{uploaded_file.name}' 파일이 업로드되었습니다!")
    
    st.markdown("---")
    st.header("💡 교육적 기능")
    
    # 교육적 기능 1: 코드 설명 기능
    explain_code = st.toggle("생성된 코드 설명 보기")
    
    # 교육적 기능 2: 다음 단계 제안
    suggest_next = st.toggle("다음 분석 단계 제안받기")
    
    st.markdown("---")
    st.info("""
    **사용 방법:**
    1. 분석할 데이터 파일을 업로드합니다.
    2. 데이터가 표시되면, 채팅창에 원하는 분석 작업을 한글로 입력합니다. (예: '데이터 처음 5줄 보여줘')
    3. 생성된 코드를 복사하여 실제 앱에 적용해보세요!
    """)

# 파일이 업로드 되었을 때의 로직
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.header("📄 업로드된 데이터 정보")
        with st.expander("데이터 미리보기 (상위 5개 행)"):
            st.dataframe(df.head())
        
        # 데이터프레임의 기본 정보를 텍스트로 정리 (Gemini에게 전달할 정보)
        df_info = f"""
        - 파일명: {uploaded_file.name}
        - 행의 수: {df.shape[0]}
        - 열의 수: {df.shape[1]}
        - 컬럼명: {df.columns.tolist()}
        - 데이터 타입: \n{df.dtypes.to_string()}
        """
        st.text_area("데이터 요약 정보 (AI 참조용)", df_info, height=200)

        st.markdown("---")
        st.header("💬 채팅으로 코드 생성하기")

        # 채팅 기록을 위한 세션 상태 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 어떤 분석을 도와드릴까요?"}]

        # 이전 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("예: '결측치가 얼마나 있는지 알려줘'"):
            if not GEMINI_API_AVAILABLE:
                st.error("Gemini API가 준비되지 않아 채팅을 진행할 수 없습니다.")
            else:
                # 사용자 메시지 기록 및 표시
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # AI 응답 생성 및 표시
                with st.chat_message("assistant"):
                    with st.spinner("AI가 코드를 생성하는 중입니다..."):
                        response_stream = get_gemini_response(prompt, df_info)
                        
                        full_response = ""
                        response_placeholder = st.empty()
                        for chunk in response_stream:
                            # 응답 텍스트에서 코드 블록 마커 제거
                            clean_chunk = chunk.text.replace("```python", "").replace("```", "")
                            full_response += clean_chunk
                            response_placeholder.code(full_response, language="python")
                        
                        # 완성된 코드 블록 표시
                        response_placeholder.code(full_response, language="python")
                
                # AI 응답 기록
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # 교육적 기능: 코드 설명
                if explain_code and full_response:
                    with st.expander("코드 설명 보기 🔍"):
                        with st.spinner("AI가 코드를 설명하는 중입니다..."):
                            explain_prompt = f"다음 Python 코드가 어떤 역할을 하는지 각 줄별로 초보자가 이해하기 쉽게 설명해줘:\n\n```python\n{full_response}\n```"
                            explain_response = get_gemini_response(explain_prompt, df_info)
                            st.write_stream(explain_response)
                
                # 교육적 기능: 다음 단계 제안
                if suggest_next and full_response:
                    with st.expander("다음 분석 단계 추천 💡"):
                         with st.spinner("AI가 다음 단계를 제안하는 중입니다..."):
                            suggestion_prompt = f"방금 '{prompt}' 요청에 따라 코드를 생성했습니다. 이 분석에 이어 시도해볼 만한 흥미로운 다음 데이터 분석 질문 3가지를 추천해주세요. 질문만 간결하게 리스트 형태로 보여주세요."
                            suggestion_response = get_gemini_response(suggestion_prompt, df_info)
                            st.write_stream(suggestion_response)

else:
    st.info("데이터 분석을 시작하려면 사이드바에서 파일을 업로드해주세요.")
