import os
import boto3
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# =========================
#      CONFIG LOADER
# =========================

def load_config():
    config = {
        "AWS_ACCESS_KEY": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": os.getenv("AWS_REGION"),
        "FILE_PATH": "s3://audio-mom/tneb-official-doc.pdf",
        "PERSIST_DIR": "db_textract",
        "CHUNK_SIZE": 600,
        "CHUNK_OVERLAP": 100
    }

    missing = [k for k,v in config.items() if not v and "AWS" in k]
    if missing:
        raise ValueError(f"Missing environment values: {missing}")

    return config


# =========================
#   AWS CLIENT INITIALIZER
# =========================

def init_aws_clients(config):
    session = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=config["AWS_ACCESS_KEY"],
        aws_secret_access_key=config["AWS_SECRET_KEY"],
        region_name=config["AWS_REGION"]
    )
    print(" AWS Authentication Successful")

    textract_client = boto3.client(
        "textract",
        aws_access_key_id=config["AWS_ACCESS_KEY"],
        aws_secret_access_key=config["AWS_SECRET_KEY"],
        region_name=config["AWS_REGION"]
    )

    return session, textract_client


# =========================
#     PDF LOADING + SPLIT
# =========================

def load_and_split_pdf(textract_client, config):
    loader = AmazonTextractPDFLoader(config["FILE_PATH"], client=textract_client)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {config['FILE_PATH']}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["CHUNK_SIZE"],
        chunk_overlap=config["CHUNK_OVERLAP"]
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} text chunks")
    return chunks


# =========================
#      VECTOR DB LOADER
# =========================

def init_vector_db(chunks, embeddings, config):
    if os.path.exists(config["PERSIST_DIR"]):
        db = Chroma(
            persist_directory=config["PERSIST_DIR"],
            embedding_function=embeddings
        )
        print("Loaded existing Vector DB")
    else:
        db = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=config["PERSIST_DIR"]
        )
        print("Vector DB created & saved.")

    return db


# =========================
#   SEARCH PDF FUNCTION
# =========================

def search_pdf(db, query: str):
    results = db.max_marginal_relevance_search(
        query=query, k=3, fetch_k=20, lambda_mult=0.5
    )

    output = "\n--- Retrieved Context ---\n"
    for r in results:
        output += r.page_content + "\n"

    return output


# =========================
#     LLM CHAIN BUILDER
# =========================

def build_chain(session, db):
    llm = ChatBedrock(
        model="meta.llama3-70b-instruct-v1:0",
        temperature=0,
        client=session
    )

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
Answer like you are the first person in this conversation. You are a government policy advisor. 
Your task is to explain details, in a professional but simple manner, as if assisting a citizen.
Always base your answers strictly on the provided context.

When responding:
1)Be polite, helpful, and factual
2)Summarize only what is relevant
3)If context lacks details, say: “I don’t have that information available right now..”
4)Read the full context before answering.
5)Keep answers concise (2–4 sentences)
6)Quote specific details when available.

Context: {context}

Question: {question}
"""
    )

    chain = RunnableSequence(
        {
            "question": RunnablePassthrough(),
            "context": lambda q: search_pdf(db, q)
        }
        | prompt
        | llm
    )

    return chain


# =========================
#      MAIN HANDLER
# =========================

def ask_chatbot(question: str):
    response = chain.invoke(question)
    return response.content if hasattr(response, "content") else response


# =========================
#          MAIN
# =========================

if __name__ == "__main__":
    config = load_config()
    session, textract_client = init_aws_clients(config)
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    if os.path.exists(config["PERSIST_DIR"]):
        print("Vector DB exists → Skipping Textract, chunking, embedding...")
        chunks = None  # Chunks not needed if DB exists
    else:
        chunks = load_and_split_pdf(textract_client, config)

    db = init_vector_db(chunks, embeddings, config)
    chain = build_chain(session, db)

    print("\n RAG PDF Chatbot Ready!")

    while True:
        query = input("\nAsk something (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = ask_chatbot(query)
        print("\n Answer:", answer)
