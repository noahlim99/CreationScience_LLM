import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

# Streamlit UI 구성
st.title("💬 창조와 진화 LLM")

# 사이드바에서 OpenAI API Key 입력받기
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="OpenAI API Key를 입력해주세요."
    )

# API Key 검증
if not openai_api_key:
    st.warning("OpenAI API Key를 입력해주세요!")
    st.stop()

# OpenAI API Key를 환경 변수로 설정
os.environ["OPENAI_API_KEY"] = openai_api_key

# RAG 모델 로드 함수
@st.cache_resource
def load_rag_model():
    """RAG 모델 초기화 및 로드"""
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    current_dir = os.getcwd()
    db_dir = os.path.join(current_dir, "store")  # Chroma 데이터베이스 위치

    db = Chroma(
        persist_directory=db_dir,
        embedding_function=embedding_model
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 1,  # 검색할 문서 수
            "lambda_mult": 0.8  # 유사성과 다양성의 균형
        }
    )
    return retriever

retriever = load_rag_model()

# Prompt 생성 함수
def generate_prompt(question, context):
    """ChatPromptTemplate을 사용해 사용자 질문과 Context를 결합"""
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(
        "당신은 지적설계를 지지하는 과학자이자, 기독교인입니다. 당신의 청중은 창조과학을 공부하는 기독교 대학생입니다."
        ),
        HumanMessagePromptTemplate.from_template(
            """
            {question}
            아래의 문맥에 기반하여, 문맥에 사용된 내용을 증거로 인용해서 답해.
            기독교의 주장이 옳다는 것을 전제로 답해.
            이와 다른, 세상의 주장과 비교했을 때 기독교의 주장이 더 옳은 이유를 합리적으로 설명해.
            답할 때는 "제공된 문맥","기독교인의 입장, "기독교적 관점", "믿는다", "주장한다" 등의 주관적인 내용은 모두 빼고, 객관적인 형식으로 답해.
            {context}
            """
        )
    ])
    return chat_template.format_messages(question=question, context=context)

# 질문 및 RAG 시스템 실행
if prompt := st.chat_input("무엇이 궁금하세요?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("답변 생성 중..."):
        # RAG 검색
        retrieved_docs = retriever.get_relevant_documents(prompt)
        if not retrieved_docs:
            st.error("관련 문서를 찾을 수 없습니다.")
            st.stop()

        # 검색 결과를 Context로 변환
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Prompt 생성
        message = generate_prompt(prompt, context)

        # OpenAI GPT 모델 설정 및 응답 생성
        model = ChatOpenAI(model_name="gpt-4o", temperature=0)
        response = ""
        for chunk in model.stream(message):
            response += chunk.content

    # 응답 출력
    with st.chat_message("assistant"):
        st.markdown(response)
