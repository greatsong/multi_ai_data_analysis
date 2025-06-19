import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import json
import base64
from datetime import datetime
import chardet
import time
import re

# OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Google Gemini
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Anthropic Claude
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 교육 도우미",
    page_icon="📊",
    layout="wide"
)

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'generated_codes' not in st.session_state:
    st.session_state.generated_codes = []
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []

# AI 모델 설정
def get_ai_response(prompt, api_key, model_type, model_name, system_prompt=""):
    try:
        if not api_key:
            return "API 키가 입력되지 않았습니다. 사이드바에서 API 키를 입력해주세요."
            
        if model_type == "OpenAI" and OpenAI:
            try:
                client = OpenAI(api_key=api_key)
                messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
                messages.append({"role": "user", "content": prompt})
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    timeout=30
                )
                return response.choices[0].message.content
            except Exception as e:
                if "api_key" in str(e).lower():
                    return "OpenAI API 키가 유효하지 않습니다."
                return f"OpenAI 오류: {str(e)}"
            
        elif model_type == "Gemini" and genai:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                if "api_key" in str(e).lower():
                    return "Gemini API 키가 유효하지 않습니다."
                return f"Gemini 오류: {str(e)}"
            
        elif model_type == "Claude" and Anthropic:
            try:
                client = Anthropic(api_key=api_key)
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": full_prompt}],
                    timeout=30
                )
                return response.content[0].text
            except Exception as e:
                if "api_key" in str(e).lower():
                    return "Claude API 키가 유효하지 않습니다."
                return f"Claude 오류: {str(e)}"
            
    except Exception as e:
        return f"예상치 못한 오류 발생: {str(e)}"
    
    return f"{model_type}의 라이브러리가 설치되지 않았습니다."

# 데이터 분석 함수
def analyze_data(df):
    analysis = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        "sample_data": df.head(5).to_dict(),
        "unique_counts": {col: df[col].nunique() for col in df.columns}
    }
    return analysis

# 파일 다운로드 함수
def get_download_link(df, filename, file_format='csv', encoding='utf-8-sig'):
    if file_format == 'csv':
        csv = df.to_csv(index=False, encoding=encoding)
        b64 = base64.b64encode(csv.encode(encoding)).decode()
        mime = 'text/csv'
    else:  # excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(output.getvalue()).decode()
        mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">📥 {filename} 다운로드</a>'
    return href

# 코드 추출 함수
def extract_code_from_response(response):
    # 코드 블록 찾기
    code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
    return code_blocks

# 메인 앱
st.title("🎓 데이터 분석 교육 도우미")
st.markdown("### AI와 함께 데이터를 탐색하고 분석해보세요!")

# 사이드바 - AI 설정
with st.sidebar:
    st.header("⚙️ AI 설정")
    
    # AI 모델 선택
    model_type = st.selectbox(
        "AI 모델 선택",
        ["OpenAI", "Gemini", "Claude"]
    )
    
    # 모델별 세부 옵션
    if model_type == "OpenAI":
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    elif model_type == "Gemini":
        model_options = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
    else:  # Claude
        model_options = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
    
    model_name = st.selectbox("모델 버전", model_options)
    
    # API 키 입력
    api_key = st.text_input(
        f"{model_type} API 키",
        type="password",
        placeholder="API 키를 입력하세요"
    )
    
    # 기본 API 키 설정
    if not api_key:
        try:
            if model_type == "OpenAI":
                api_key = st.secrets.get("openai_api_key", "")
            elif model_type == "Gemini":
                api_key = st.secrets.get("gemini_api_key", "")
            else:
                api_key = st.secrets.get("claude_api_key", "")
            
            if api_key:
                st.success("✅ 기본 API 키 로드됨")
        except:
            st.info("💡 API 키를 입력해주세요")
    
    st.divider()
    
    # 대화 컨텍스트 관리
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        st.session_state.generated_codes = []
        st.rerun()
    
    st.divider()
    
    # requirements.txt 생성
    if st.button("📋 requirements.txt 생성"):
        requirements = """streamlit
pandas
numpy
plotly
openpyxl
xlsxwriter
openai
google-generativeai
anthropic
chardet"""
        st.code(requirements, language="text")

# 메인 컨텐츠
col1, col2 = st.columns([1, 2])

# 왼쪽: 데이터 업로드
with col1:
    st.header("📤 데이터 업로드")
    
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 선택하세요",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        try:
            # 파일 읽기 with 인코딩 자동 감지
            if uploaded_file.name.endswith('.csv'):
                encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
                df = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ {encoding} 인코딩으로 읽기 성공")
                        break
                    except:
                        continue
                
                if df is None:
                    uploaded_file.seek(0)
                    raw_data = uploaded_file.read()
                    detected = chardet.detect(raw_data)
                    st.error(f"인코딩 자동 감지 실패 (감지: {detected['encoding']})")
            else:
                df = pd.read_excel(uploaded_file)
                st.success("✅ Excel 파일 읽기 성공")
            
            if df is not None:
                st.session_state.df = df
                
                # 데이터 미리보기
                st.subheader("📋 데이터 미리보기")
                st.dataframe(df.head(5), height=200)
                
                # 기본 정보
                analysis = analyze_data(df)
                st.metric("행 수", analysis['shape'][0])
                st.metric("열 수", analysis['shape'][1])
                
                # 첫 업로드 시 자동 분석 메시지 추가
                if len(st.session_state.messages) == 0:
                    intro_message = f"""데이터가 업로드되었습니다! 

📊 **데이터 개요:**
- 크기: {analysis['shape'][0]}행 × {analysis['shape'][1]}열
- 열: {', '.join(analysis['columns'][:5])}{'...' if len(analysis['columns']) > 5 else ''}

어떤 분석을 하고 싶으신가요? 예를 들어:
- "이 데이터의 특성을 분석해줘"
- "시간대별 추이를 보고 싶어"
- "카테고리별 비교 그래프를 만들어줘"
- "이상치를 찾아서 처리하고 싶어"

무엇이든 물어보세요! 제가 적절한 코드를 만들어드릴게요. 😊"""
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": intro_message
                    })
                
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")
    
    # 데이터 다운로드 섹션
    if st.session_state.df_processed is not None:
        st.divider()
        st.subheader("📥 처리된 데이터 다운로드")
        
        file_format = st.selectbox("형식", ["CSV", "Excel"], key="download_format")
        
        if file_format == "CSV":
            encoding_option = st.selectbox(
                "인코딩", 
                ["utf-8-sig (권장)", "cp949", "euc-kr"],
                key="download_encoding"
            )
            encoding_map = {
                "utf-8-sig (권장)": "utf-8-sig",
                "cp949": "cp949",
                "euc-kr": "euc-kr"
            }
            selected_encoding = encoding_map[encoding_option]
            
            st.markdown(
                get_download_link(
                    st.session_state.df_processed, 
                    "processed_data.csv", 
                    "csv", 
                    selected_encoding
                ), 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                get_download_link(
                    st.session_state.df_processed, 
                    "processed_data.xlsx", 
                    "excel"
                ), 
                unsafe_allow_html=True
            )

# 오른쪽: AI 대화
with col2:
    st.header("💬 AI와 데이터 분석 대화")
    
    # 대화 내역 표시
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # 코드가 포함된 경우 실행 버튼 추가
                if message["role"] == "assistant" and "```python" in message["content"]:
                    code_blocks = extract_code_from_response(message["content"])
                    for idx, code in enumerate(code_blocks):
                        col_run, col_copy = st.columns([1, 1])
                        with col_run:
                            if st.button(f"▶️ 코드 실행", key=f"run_{len(st.session_state.messages)}_{idx}"):
                                try:
                                    # 안전한 실행 환경
                                    exec_globals = {
                                        'st': st,
                                        'pd': pd,
                                        'np': np,
                                        'px': px,
                                        'go': go,
                                        'df': st.session_state.df,
                                        'df_processed': st.session_state.df_processed
                                    }
                                    
                                    # 실행 결과를 캡처하기 위한 컨테이너
                                    with st.expander("실행 결과", expanded=True):
                                        exec(code, exec_globals, exec_globals)
                                        
                                        # df_processed가 업데이트되었는지 확인
                                        if 'df_processed' in exec_globals and exec_globals['df_processed'] is not None:
                                            st.session_state.df_processed = exec_globals['df_processed']
                                            
                                except Exception as e:
                                    st.error(f"실행 오류: {str(e)}")
                        
                        with col_copy:
                            if st.button(f"📋 복사", key=f"copy_{len(st.session_state.messages)}_{idx}"):
                                st.code(code, language="python")
                                st.info("위 코드를 복사하세요!")
    
    # 사용자 입력
    user_input = st.chat_input("데이터에 대해 궁금한 점을 자유롭게 물어보세요!")
    
    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI 응답 생성
        if st.session_state.df is not None:
            # 시스템 프롬프트
            system_prompt = """당신은 친절한 데이터 분석 교육 도우미입니다. 
사용자와 자연스럽게 대화하며 데이터 분석을 도와주세요.

중요 규칙:
1. 사용자의 의도를 파악하여 적절한 분석 방법과 시각화를 제안하세요
2. 코드가 필요한 경우, 자연스럽게 "이런 코드를 만들어드릴까요?"라고 물어보세요
3. 코드를 생성할 때는 반드시 ```python 코드블록 ``` 형식을 사용하세요
4. 한글 주석으로 코드를 설명하세요
5. plotly를 사용하여 인터랙티브한 시각화를 만드세요
6. 실행 가능한 완전한 코드를 제공하세요
7. df는 이미 로드되어 있으므로 df를 직접 사용하세요
8. 전처리된 데이터는 df_processed로 저장하세요"""

            # 데이터 컨텍스트 생성
            analysis = analyze_data(st.session_state.df)
            data_context = f"""
현재 데이터 정보:
- 크기: {analysis['shape']}
- 열: {list(analysis['columns'])}
- 데이터 타입: {analysis['dtypes']}
- 결측값: {analysis['missing_values']}
- 고유값 개수: {analysis['unique_counts']}
"""

            # 대화 히스토리 포함
            conversation_history = "\n".join([
                f"{msg['role']}: {msg['content'][:200]}..." 
                for msg in st.session_state.messages[-5:]  # 최근 5개 대화
            ])

            # 전체 프롬프트
            full_prompt = f"""
{data_context}

최근 대화 내용:
{conversation_history}

사용자 질문: {user_input}

위 맥락을 고려하여 자연스럽고 도움이 되는 응답을 해주세요.
필요시 실행 가능한 코드를 제안하세요."""

            # AI 응답 받기
            with st.spinner("생각 중..."):
                response = get_ai_response(
                    full_prompt, 
                    api_key, 
                    model_type, 
                    model_name,
                    system_prompt
                )
            
            # 응답 추가
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        else:
            # 데이터가 없는 경우
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "먼저 데이터를 업로드해주세요! 왼쪽에서 CSV나 Excel 파일을 선택하실 수 있어요. 📊"
            })
        
        # 페이지 새로고침
        st.rerun()

# 하단 정보
st.divider()
with st.expander("💡 사용 팁", expanded=False):
    st.markdown("""
    ### 이런 질문을 해보세요:
    - "이 데이터의 전체적인 특성을 분석해줘"
    - "A열과 B열의 상관관계를 보여줘"
    - "시간대별 변화 추이를 그래프로 그려줘"
    - "카테고리별로 그룹화해서 평균을 계산해줘"
    - "이상치를 찾아서 제거하는 방법을 알려줘"
    - "머신러닝을 위한 전처리 방법을 제안해줘"
    
    ### 코드 실행:
    - AI가 생성한 코드는 바로 실행할 수 있습니다
    - 실행 결과는 바로 확인 가능합니다
    - 처리된 데이터는 다운로드할 수 있습니다
    """)

# 생성된 코드 모음 (사이드바 하단)
with st.sidebar:
    if len(st.session_state.generated_codes) > 0:
        st.divider()
        st.header("📝 생성된 코드 모음")
        for idx, code_info in enumerate(st.session_state.generated_codes):
            with st.expander(f"코드 {idx+1}: {code_info['title'][:20]}..."):
                st.code(code_info['code'], language="python")
                if st.button(f"📋 복사", key=f"sidebar_copy_{idx}"):
                    st.info("코드를 복사하세요!")
