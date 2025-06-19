import streamlit as st
import pandas as pd
import json
import re
from io import StringIO
import requests
import google.generativeai as genai
from openai import OpenAI
import anthropic

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 스트림릿 코드 생성기",
    page_icon="📊",
    layout="wide"
)

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = []
if 'data_info' not in st.session_state:
    st.session_state.data_info = None

# 타이틀
st.title("📊 데이터 분석 스트림릿 코드 생성기")
st.markdown("데이터를 업로드하고 AI와 대화하며 스트림릿 코드를 생성하세요!")

# 사이드바 - API 설정
with st.sidebar:
    st.header("⚙️ API 설정")
    
    api_provider = st.selectbox(
        "AI 제공자 선택",
        ["Claude (Anthropic)", "Gemini (Google)", "ChatGPT (OpenAI)"]
    )
    
    # API 키 입력 (시크릿 사용 권장)
    st.info("💡 프로덕션 환경에서는 Streamlit Secrets를 사용하세요")
    
    if api_provider == "Claude (Anthropic)":
        api_key = st.text_input(
            "Claude API Key",
            value=st.secrets.get("CLAUDE_API_KEY", ""),
            type="password"
        )
    elif api_provider == "Gemini (Google)":
        api_key = st.text_input(
            "Gemini API Key",
            value=st.secrets.get("GEMINI_API_KEY", ""),
            type="password"
        )
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.secrets.get("OPENAI_API_KEY", ""),
            type="password"
        )
    
    st.divider()
    
    # 학습 도우미 기능
    st.header("📚 학습 도우미")
    show_explanation = st.checkbox("코드 설명 포함", value=True)
    show_concepts = st.checkbox("데이터 분석 개념 설명", value=True)
    difficulty_level = st.select_slider(
        "난이도 선택",
        options=["초급", "중급", "고급"],
        value="중급"
    )

# 메인 컨텐츠를 두 열로 나누기
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 데이터 업로드")
    
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 선택하세요",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        # 파일 읽기
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ 파일 업로드 성공: {uploaded_file.name}")
            
            # 데이터 미리보기
            st.subheader("데이터 미리보기")
            st.dataframe(df.head())
            
            # 데이터 정보 저장
            data_info = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "sample": df.head().to_dict(),
                "description": df.describe().to_dict() if df.select_dtypes(include=['number']).shape[1] > 0 else None
            }
            st.session_state.data_info = data_info
            
            # 데이터 정보 표시
            with st.expander("📊 데이터 정보"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("행 수", data_info["shape"][0])
                    st.metric("열 수", data_info["shape"][1])
                with col_b:
                    st.write("**컬럼 목록:**")
                    for col in data_info["columns"]:
                        st.write(f"- {col} ({data_info['dtypes'][col]})")
        
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")

with col2:
    st.header("💬 AI 채팅")
    
    # 채팅 컨테이너
    chat_container = st.container()
    
    # 사용자 입력
    user_input = st.chat_input("데이터 분석에 대해 물어보세요 (예: '판매량 추이를 보여주는 차트를 만들어줘')")
    
    if user_input and api_key:
        # 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI 응답 생성
        with st.spinner("AI가 코드를 생성 중입니다..."):
            try:
                # 프롬프트 생성
                prompt = create_prompt(user_input, st.session_state.data_info, 
                                     show_explanation, show_concepts, difficulty_level)
                
                # API 호출
                if api_provider == "Claude (Anthropic)":
                    response = get_claude_response(api_key, prompt)
                elif api_provider == "Gemini (Google)":
                    response = get_gemini_response(api_key, prompt)
                else:
                    response = get_openai_response(api_key, prompt)
                
                # 응답 추가
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # 코드 추출 및 저장
                code = extract_code(response)
                if code:
                    st.session_state.generated_code.append({
                        "request": user_input,
                        "code": code,
                        "explanation": response
                    })
                
            except Exception as e:
                st.error(f"AI 응답 생성 중 오류: {str(e)}")
    
    # 채팅 히스토리 표시
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# 생성된 코드 섹션
st.divider()
st.header("📝 생성된 코드")

if st.session_state.generated_code:
    # 탭으로 코드들 구분
    tabs = st.tabs([f"코드 {i+1}" for i in range(len(st.session_state.generated_code))])
    
    for i, (tab, code_item) in enumerate(zip(tabs, st.session_state.generated_code)):
        with tab:
            st.subheader(f"요청: {code_item['request']}")
            
            # 코드 표시 및 복사 버튼
            col1, col2 = st.columns([10, 1])
            with col1:
                st.code(code_item['code'], language='python')
            with col2:
                if st.button("📋", key=f"copy_{i}", help="코드 복사"):
                    st.write("클립보드에 복사됨!")
                    st.session_state[f"copied_{i}"] = True
            
            # 설명 표시
            if show_explanation:
                with st.expander("💡 설명 보기"):
                    st.write(code_item['explanation'])

# 하단 정보
st.divider()
st.info("""
### 💡 사용 팁
1. **데이터 업로드**: CSV 또는 Excel 파일을 업로드하세요
2. **질문하기**: 원하는 분석이나 시각화를 자연어로 요청하세요
3. **코드 복사**: 생성된 코드를 복사하여 새 스트림릿 앱에서 사용하세요
4. **학습하기**: 코드 설명과 개념 설명을 통해 데이터 분석을 배워보세요

### 📚 추천 학습 자료
- [Streamlit 공식 문서](https://docs.streamlit.io)
- [Pandas 튜토리얼](https://pandas.pydata.org/docs/getting_started/tutorials.html)
- [Plotly 차트 갤러리](https://plotly.com/python/)
""")

# 헬퍼 함수들
def create_prompt(user_input, data_info, show_explanation, show_concepts, difficulty_level):
    """AI에게 전달할 프롬프트 생성"""
    prompt = f"""
    사용자가 Streamlit 앱을 위한 데이터 분석 코드를 요청했습니다.
    
    사용자 요청: {user_input}
    
    데이터 정보:
    - 크기: {data_info['shape'] if data_info else '데이터 없음'}
    - 컬럼: {data_info['columns'] if data_info else '데이터 없음'}
    - 데이터 타입: {data_info['dtypes'] if data_info else '데이터 없음'}
    
    요구사항:
    1. 완전하고 실행 가능한 Streamlit 코드를 생성하세요
    2. 필요한 모든 import 문을 포함하세요
    3. 에러 처리를 포함하세요
    4. 난이도: {difficulty_level}
    """
    
    if show_explanation:
        prompt += "\n5. 코드의 각 부분을 설명하는 주석을 포함하세요"
    
    if show_concepts:
        prompt += "\n6. 사용된 데이터 분석 개념을 간단히 설명하세요"
    
    return prompt

def extract_code(response):
    """응답에서 Python 코드 추출"""
    # 코드 블록 찾기
    code_pattern = r'```python\n(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 백틱 없이 코드만 있는 경우
    if "import" in response and "st." in response:
        return response.strip()
    
    return None

def get_claude_response(api_key, prompt):
    """Claude API 호출"""
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

def get_gemini_response(api_key, prompt):
    """Gemini API 호출"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    response = model.generate_content(prompt)
    return response.text

def get_openai_response(api_key, prompt):
    """OpenAI API 호출"""
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    
    return response.choices[0].message.content
