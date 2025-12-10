import os
import boto3
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_chroma import Chroma
from chromadb import Client as ChromaClient
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

session = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

print(" AWS Authentication Successful")

textract_client = boto3.client("textract", region_name="ap-south-1")

file_path = "s3://audio-mom/tneb-official-doc.pdf"
loader = AmazonTextractPDFLoader(file_path, client=textract_client)
documents = loader.load()

print(f"Loaded {len(documents)} documents from {file_path}")

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
PERSIST_DIR = "db_textract"

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} text chunks")



if os.path.exists(PERSIST_DIR):
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    print("Loaded existing Vector DB")
else:
    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    #db.persist()
    print("Vector DB created & saved.")

def search_pdf(query: str):
    """Searches the stored document and returns best matching text."""
    results = db.max_marginal_relevance_search(query=query, k=3, fetch_k=20, lambda_mult=0.5)

    output = "\n--- Retrieved Context ---\n"
    for r in results:
        output += r.page_content + "\n"

    return output

llm = ChatBedrock(
    model="meta.llama3-70b-instruct-v1:0",
    temperature=0,
    client=session
)

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
Answer like you are the first person in this conversation. You are a government policy advisor. Your task is to explain details, in a professional but simple manner, as if assisting a citizen.
Always base your answers strictly on the provided context.
Do not include information from outside the PDF.

When responding:
1)Be polite, helpful, and factual
2)Summarize only what is relevant
3)If context lacks details, say: “I don’t have that information available right now..”
4)Read the full context before answering.
5)Keep answers concise (2–4 sentences)
6)Quote specific details when available.
Format:

Context: {context}

Question: {question}

"""
)

chain = RunnableSequence(
    {
        "question": RunnablePassthrough(),
        "context": search_pdf
    }
    | prompt
    | llm
)


# =========================
#        Main Handler
# =========================

def ask_chatbot(question: str):
    """Use this tool for all government scheme information."""
    response = chain.invoke(question)
    return response.content if hasattr(response, "content") else response


# =========================
#        CLI Mode
# =========================

if __name__ == "__main__":
    print("\n RAG PDF Chatbot Ready!")
    while True:
        query = input("\nAsk something (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = ask_chatbot(query)
        print("\n Answer:", answer)
