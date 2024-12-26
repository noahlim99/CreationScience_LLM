import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

# 환경 변수에서 API Key 불러오기
api_key = os.environ.get("OPENAI_API_KEY")

# OpenAI Embedding 모델 설정
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 현재 디렉토리 및 벡터 스토어 경로 설정
current_dir = os.getcwd()
db_dir = os.path.join(current_dir, "store")

# Chroma 벡터 스토어 로드
db = Chroma(
    persist_directory=db_dir,
    embedding_function=embedding_model
)

# MMR 기반 검색 설정 retriver
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 1,             # 반환할 문서 수
        "lambda_mult": 0.8  # 유사성과 다양성 
    }
)

# 질문 정의
question = "빅뱅이론과 연대측정에 대해 설명해줘."

# 문서 검색 및 Context 생성
retrieved_docs = retriever.get_relevant_documents(question)
if not retrieved_docs:
    print("관련 문서를 찾을 수 없습니다.")
    exit()

context = "\n".join([doc.page_content for doc in retrieved_docs])
print(f"\n[검색된 문서 Context]\n{context[:500]}...\n")

# Prompt 생성
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(
        "당신은 지적설계를 지지하는 과학자입니다. 당신의 청중은 창조과학을 공부하는 기독교 대학생입니다."
    ),
    HumanMessagePromptTemplate.from_template(
        """
        {question}
        아래의 문맥에 기반하여 답해주세요.
        {context}
        """
    )
])

message = chat_template.format_messages(
    question=question,
    context=context
)

# GPT 모델 설정
model = ChatOpenAI(
    model_name="gpt-4o",  # 사용할 모델 이름
    temperature=0         # 답변의 다양성을 조정 (0: 결정적, 1: 다양성 높음)
)

# 응답 생성 및 출력
print("\n[모델 응답]")
response = ""
for chunk in model.stream(message):
    response += chunk.content
    print(chunk.content, end="", flush=True)

print("\n")