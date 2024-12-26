import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 환경 변수에서 API Key 불러오기(가상환경에 저장)
api_key = os.environ.get("OPENAI_API_KEY")

# 현재 워킹 디렉토리 설정
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "output")
db_dir = os.path.join(current_dir, "store")

# 문서 로드, Langchain의 document객체로 변환 후 pages 리스트에 저장 
pages = []
if os.path.exists(output_dir):
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    content = data.get("content", "")
                    if content.strip():
                        pages.append(Document(page_content=content))
                except json.JSONDecodeError:
                    print(f"파일 {file_name}은 올바른 JSON 형식이 아닙니다.")
else:
    raise FileNotFoundError(f"{output_dir} 디렉토리가 존재하지 않습니다.")
print(f"{len(pages)}개 문서 처리 완료")

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)
print(f"{len(docs)}개의 청크로 분할 완료")

# 벡터 스토어 생성
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
db = Chroma.from_documents(docs, embedding=embedding_model, persist_directory=db_dir)
print(f"Chroma 벡터 스토어 저장 완료: {db_dir}")

# MMR 기반 검색 retriever
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 1, "lambda_mult": 0.8})

# 검색 테스트
question = "빅뱅이론과 연대측정에 대해 설명해줘."
retrieved_docs = retriever.get_relevant_documents(question)
if retrieved_docs:
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    print(f"검색된 문서 개수: {len(retrieved_docs)}")
    print(f"생성된 컨텍스트 일부:\n{context[:500]}")
else:
    print("관련 문서를 찾을 수 없습니다.")