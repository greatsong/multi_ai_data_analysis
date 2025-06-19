import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import json
import base64
from datetime import datetime
import chardet  # 인코딩 감지용

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
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = None

# AI 모델 설정
def get_ai_response(prompt, api_key, model_type, model_name):
    try:
        if not api_key:
            return "API 키가 입력되지 않았습니다. 사이드바에서 API 키를 입력해주세요."
            
        if model_type == "OpenAI" and OpenAI:
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    timeout=30  # 30초 타임아웃
                )
                return response.choices[0].message.content
            except Exception as e:
                if "api_key" in str(e).lower():
                    return "OpenAI API 키가 유효하지 않습니다. 올바른 키인지 확인해주세요."
                elif "connection" in str(e).lower():
                    return "OpenAI 서버 연결 오류입니다. 잠시 후 다시 시도해주세요."
                else:
                    return f"OpenAI 오류: {str(e)}"
            
        elif model_type == "Gemini" and genai:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                if "api_key" in str(e).lower() or "API_KEY_INVALID" in str(e):
                    return "Gemini API 키가 유효하지 않습니다. 올바른 키인지 확인해주세요."
                elif "connection" in str(e).lower():
                    return "Gemini 서버 연결 오류입니다. 잠시 후 다시 시도해주세요."
                else:
                    return f"Gemini 오류: {str(e)}"
            
        elif model_type == "Claude" and Anthropic:
            try:
                client = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30  # 30초 타임아웃
                )
                return response.content[0].text
            except Exception as e:
                if "api_key" in str(e).lower() or "authentication" in str(e).lower():
                    return "Claude API 키가 유효하지 않습니다. 올바른 키인지 확인해주세요."
                elif "connection" in str(e).lower():
                    return "Claude 서버 연결 오류입니다. 잠시 후 다시 시도해주세요."
                else:
                    return f"Claude 오류: {str(e)}"
            
    except Exception as e:
        return f"예상치 못한 오류 발생: {str(e)}"
    
    return f"{model_type}의 라이브러리가 설치되지 않았습니다. requirements.txt를 확인해주세요."

# 데이터 분석 함수
def analyze_data(df):
    analysis = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        "sample_data": df.head(5).to_dict()
    }
    return analysis

# 파일 다운로드 함수
def get_download_link(df, filename, file_format='csv', encoding='utf-8-sig'):
    if file_format == 'csv':
        # utf-8-sig는 엑셀에서 한글이 깨지지 않도록 BOM을 추가
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

# 메인 앱
st.title("🎓 데이터 분석 교육 도우미")
st.markdown("### 고등학생과 선생님을 위한 AI 기반 데이터 분석 도구")

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
        placeholder="API 키를 입력하세요 (선택사항)",
        help="API 키를 입력하지 않으면 기본 키가 사용됩니다 (제한적)"
    )
    
    # 기본 API 키 설정 (실제 사용시에는 환경변수로 관리)
    if not api_key:
        # 여기에 기본 API 키를 설정할 수 있습니다
        # api_key = st.secrets.get(f"{model_type.lower()}_api_key", "")
        st.info("기본 API 키를 사용합니다 (제한적 사용)")
    
    st.divider()
    
    # 교육 도구
    st.header("📚 교육 도구")
    show_data_tips = st.checkbox("데이터 분석 팁 보기", value=True)
    show_code_explanation = st.checkbox("코드 설명 추가", value=True)
    
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
        st.info("위 내용을 복사하여 requirements.txt 파일에 붙여넣으세요")

# 메인 컨텐츠
tab1, tab2, tab3, tab4 = st.tabs(["📤 데이터 업로드", "💬 데이터 분석 대화", "📊 시각화 코드 생성", "📖 학습 자료"])

# 탭 1: 데이터 업로드
with tab1:
    st.header("1단계: 데이터 업로드 및 전처리")
    
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 업로드하세요",
        type=['csv', 'xlsx', 'xls'],
        help="파일 크기는 200MB 이하로 제한됩니다"
    )
    
    if uploaded_file is not None:
        try:
            # 파일 읽기 with 인코딩 자동 감지
            if uploaded_file.name.endswith('.csv'):
                # CSV 파일의 인코딩 자동 감지
                encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
                df = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # 파일 포인터 초기화
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ 파일을 {encoding} 인코딩으로 성공적으로 읽었습니다.")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        continue
                
                if df is None:
                    # 더 정교한 인코딩 감지
                    uploaded_file.seek(0)
                    raw_data = uploaded_file.read()
                    detected = chardet.detect(raw_data)
                    
                    st.warning(f"자동 인코딩 감지 결과: {detected['encoding']} (신뢰도: {detected['confidence']:.1%})")
                    
                    encoding_choice = st.selectbox(
                        "인코딩을 직접 선택해주세요:",
                        ['cp949', 'euc-kr', 'utf-8', 'latin1', 'utf-16'],
                        index=0 if detected['encoding'] in ['cp949', 'euc-kr'] else 2
                    )
                    if st.button("선택한 인코딩으로 다시 시도"):
                        uploaded_file.seek(0)
                        try:
                            df = pd.read_csv(uploaded_file, encoding=encoding_choice)
                            st.success(f"✅ {encoding_choice} 인코딩으로 파일을 읽었습니다.")
                        except Exception as e:
                            st.error(f"선택한 인코딩으로도 실패: {str(e)}")
            else:
                # Excel 파일은 일반적으로 인코딩 문제가 없음
                df = pd.read_excel(uploaded_file)
                st.success("✅ Excel 파일을 성공적으로 읽었습니다.")
            
            if df is not None:
                st.session_state.df = df
            
            # 데이터 미리보기
            st.subheader("📋 데이터 미리보기")
            st.dataframe(df.head(10))
            
            # 데이터 분석
            analysis = analyze_data(df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("행 수", analysis['shape'][0])
            with col2:
                st.metric("열 수", analysis['shape'][1])
            with col3:
                st.metric("결측값 열", sum(1 for v in analysis['missing_values'].values() if v > 0))
            
            # AI 분석
            if st.button("🤖 AI로 데이터 분석하기"):
                with st.spinner("AI가 데이터를 분석 중입니다..."):
                    prompt = f"""
다음 데이터를 분석하고 전처리 방법을 제안해주세요:

데이터 정보:
- 크기: {analysis['shape']}
- 열: {', '.join(analysis['columns'])}
- 데이터 타입: {analysis['dtypes']}
- 결측값: {analysis['missing_values']}

1. 이 데이터의 특성과 용도를 추측해주세요.
2. 필요한 전처리 단계를 제안해주세요.
3. 주의해야 할 점이나 질문사항을 알려주세요.
"""
                    
                    response = get_ai_response(prompt, api_key, model_type, model_name)
                    
                    st.info("🤖 AI 분석 결과")
                    st.write(response)
                    
                    # 전처리 옵션
                    st.subheader("전처리 옵션")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        remove_nulls = st.checkbox("결측값 제거")
                        fill_nulls = st.checkbox("결측값 채우기")
                        if fill_nulls:
                            fill_method = st.selectbox("채우기 방법", ["평균", "중앙값", "최빈값", "0"])
                    
                    with col2:
                        remove_duplicates = st.checkbox("중복 행 제거")
                        standardize_columns = st.checkbox("열 이름 표준화")
                    
                    if st.button("전처리 실행"):
                        df_processed = df.copy()
                        
                        if remove_nulls:
                            df_processed = df_processed.dropna()
                        elif fill_nulls:
                            if fill_method == "평균":
                                df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
                            elif fill_method == "중앙값":
                                df_processed = df_processed.fillna(df_processed.median(numeric_only=True))
                            elif fill_method == "최빈값":
                                df_processed = df_processed.fillna(df_processed.mode().iloc[0])
                            else:
                                df_processed = df_processed.fillna(0)
                        
                        if remove_duplicates:
                            df_processed = df_processed.drop_duplicates()
                        
                        if standardize_columns:
                            df_processed.columns = [col.strip().lower().replace(' ', '_') for col in df_processed.columns]
                        
                        st.session_state.df_processed = df_processed
                        
                        st.success("전처리가 완료되었습니다!")
                        st.dataframe(df_processed.head())
                        
                        # 다운로드 옵션
                        st.subheader("📥 다운로드 옵션")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            file_format = st.selectbox("파일 형식", ["CSV", "Excel"])
                        
                        with col2:
                            if file_format == "CSV":
                                encoding_option = st.selectbox(
                                    "인코딩", 
                                    ["utf-8-sig (권장)", "cp949", "euc-kr"],
                                    help="utf-8-sig는 엑셀에서 한글이 깨지지 않습니다"
                                )
                                encoding_map = {
                                    "utf-8-sig (권장)": "utf-8-sig",
                                    "cp949": "cp949",
                                    "euc-kr": "euc-kr"
                                }
                                selected_encoding = encoding_map[encoding_option]
                            else:
                                selected_encoding = None
                        
                        with col3:
                            filename = st.text_input("파일명", value="processed_data")
                        
                        # 다운로드 링크
                        if file_format == "CSV":
                            st.markdown(
                                get_download_link(df_processed, f"{filename}.csv", "csv", selected_encoding), 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                get_download_link(df_processed, f"{filename}.xlsx", "excel"), 
                                unsafe_allow_html=True
                            )
                        
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")

# 탭 2: 데이터 분석 대화
with tab2:
    st.header("2단계: AI와 데이터 분석 대화")
    
    if st.session_state.df is not None:
        # 채팅 인터페이스
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # 사용자 입력
        user_input = st.chat_input("데이터에 대해 궁금한 점을 물어보세요")
        
        if user_input:
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.write(user_input)
            
            # AI 응답
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    df_info = st.session_state.df.info(buf=StringIO())
                    prompt = f"""
데이터 분석 전문가로서 다음 질문에 답해주세요:

데이터 정보:
{analyze_data(st.session_state.df)}

사용자 질문: {user_input}

교육적이고 이해하기 쉽게 설명해주세요.
"""
                    
                    response = get_ai_response(prompt, api_key, model_type, model_name)
                    st.write(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("먼저 데이터를 업로드해주세요.")

# 탭 3: 시각화 코드 생성
with tab3:
    st.header("3단계: 시각화 코드 생성")
    
    if st.session_state.df is not None:
        st.subheader("어떤 시각화를 만들고 싶으신가요?")
        
        viz_type = st.selectbox(
            "시각화 유형",
            ["산점도", "막대 그래프", "선 그래프", "히스토그램", "박스 플롯", "히트맵", "파이 차트", "사용자 정의"]
        )
        
        viz_description = st.text_area(
            "시각화에 대한 설명",
            placeholder="예: 나이별 평균 점수를 막대 그래프로 보여주세요"
        )
        
        if st.button("📊 코드 생성"):
            with st.spinner("코드를 생성 중입니다..."):
                prompt = f"""
다음 데이터프레임을 사용하여 Streamlit과 Plotly로 시각화 코드를 생성해주세요:

데이터 열: {list(st.session_state.df.columns)}
데이터 타입: {st.session_state.df.dtypes.to_dict()}

요청사항:
- 시각화 유형: {viz_type}
- 설명: {viz_description}

요구사항:
1. 완전하고 실행 가능한 Streamlit 코드를 생성하세요
2. Plotly를 사용하여 인터랙티브한 그래프를 만드세요
3. 한글 레이블과 제목을 사용하세요
4. 코드에 주석을 추가하여 이해하기 쉽게 만드세요
5. 에러 처리를 포함하세요

코드만 생성하고 다른 설명은 하지 마세요.
"""
                
                code = get_ai_response(prompt, api_key, model_type, model_name)
                
                # 코드 정리
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                
                st.session_state.generated_code = code
                
                # 코드 표시
                st.subheader("생성된 코드")
                st.code(code, language="python")
                
                # 복사 버튼
                if st.button("📋 코드 복사"):
                    st.write("코드가 클립보드에 복사되었습니다!")
                    st.balloons()
                
                # 코드 실행 시도
                try:
                    st.subheader("실행 결과")
                    # 안전한 실행을 위해 제한된 환경에서 실행
                    exec_globals = {
                        'st': st,
                        'pd': pd,
                        'np': np,
                        'px': px,
                        'go': go,
                        'df': st.session_state.df
                    }
                    exec(code, exec_globals)
                except Exception as e:
                    st.error(f"코드 실행 중 오류 발생: {str(e)}")
                    st.info("생성된 코드를 확인하고 수정이 필요할 수 있습니다.")
    else:
        st.info("먼저 데이터를 업로드해주세요.")

# 탭 4: 학습 자료
with tab4:
    st.header("📖 데이터 분석 학습 자료")
    
    if show_data_tips:
        st.subheader("💡 데이터 분석 팁")
        
        tips = {
            "데이터 탐색": [
                "항상 데이터의 크기와 구조를 먼저 확인하세요",
                "결측값과 이상치를 찾아보세요",
                "각 열의 데이터 타입을 확인하세요"
            ],
            "전처리": [
                "결측값은 제거하거나 적절한 값으로 채워야 합니다",
                "중복 데이터는 분석 전에 제거하세요",
                "필요시 데이터 타입을 변환하세요"
            ],
            "시각화": [
                "목적에 맞는 그래프 유형을 선택하세요",
                "색상과 라벨을 명확하게 사용하세요",
                "인터랙티브 기능을 활용하면 더 효과적입니다"
            ]
        }
        
        for category, tip_list in tips.items():
            with st.expander(category):
                for tip in tip_list:
                    st.write(f"• {tip}")
    
    st.subheader("📊 일반적인 시각화 유형")
    
    viz_guide = {
        "산점도": "두 변수 간의 관계를 보여줄 때",
        "막대 그래프": "카테고리별 값을 비교할 때",
        "선 그래프": "시간에 따른 변화를 보여줄 때",
        "히스토그램": "데이터의 분포를 보여줄 때",
        "박스 플롯": "데이터의 분포와 이상치를 함께 볼 때",
        "히트맵": "두 카테고리 변수 간의 관계를 보여줄 때",
        "파이 차트": "전체 대비 비율을 보여줄 때"
    }
    
    for viz_type, usage in viz_guide.items():
        st.write(f"**{viz_type}**: {usage}")
    
    st.subheader("🔗 추가 학습 자료")
    st.markdown("""
    - [Pandas 공식 문서](https://pandas.pydata.org/docs/)
    - [Plotly 공식 문서](https://plotly.com/python/)
    - [Streamlit 공식 문서](https://docs.streamlit.io/)
    - [데이터 사이언스 스쿨](https://datascienceschool.net/)
    """)

# 푸터
st.divider()
st.markdown("---")
st.caption("🎓 고등학생과 선생님을 위한 데이터 분석 교육 도구 | Made with ❤️ using Streamlit")
