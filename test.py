"""Improved version of the original test1.py with better error handling and structure."""
import os
import sys
import logging
from typing import Optional
import boto3
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_environment() -> dict:
    """Validate and return environment configuration."""
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]
    config = {}
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        config[var] = value
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Optional configurations with defaults
    config["S3_BUCKET_PATH"] = os.getenv("S3_BUCKET_PATH", "s3://audio-mom/tneb-official-doc.pdf")
    config["PERSIST_DIR"] = os.getenv("PERSIST_DIR", "db_textract")
    config["CHUNK_SIZE"] = int(os.getenv("CHUNK_SIZE", "600"))
    config["CHUNK_OVERLAP"] = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    return config

def initialize_aws_clients(config: dict) -> tuple:
    """Initialize AWS clients with error handling."""
    try:
        # Bedrock session
        session = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"],
            region_name=config["AWS_REGION"]
        )
        logger.info("AWS Bedrock authentication successful")
        
        # Textract client
        textract_client = boto3.client(
            "textract",
            aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"],
            region_name=config["AWS_REGION"]
        )
        logger.info("AWS Textract client initialized")
        
        return session, textract_client
        
    except Exception as e:
        logger.error(f"Failed to initialize AWS clients: {e}")
        raise

def load_and_process_documents(textract_client, config: dict) -> list:
    """Load and process documents with error handling."""
    try:
        file_path = config["S3_BUCKET_PATH"]
        logger.info(f"Loading documents from {file_path}")
        
        loader = AmazonTextractPDFLoader(file_path, client=textract_client)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["CHUNK_SIZE"],
            chunk_overlap=config["CHUNK_OVERLAP"]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} text chunks")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to load and process documents: {e}")
        raise

def initialize_vector_db(chunks: list, session, config: dict):
    """Initialize or load vector database with error handling."""
    try:
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            client=session
        )
        
        persist_dir = config["PERSIST_DIR"]
        
        if os.path.exists(persist_dir):
            db = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            logger.info("Loaded existing Vector DB")
        else:
            db = Chroma.from_documents(
                chunks,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            logger.info("Vector DB created & saved")
        
        return db
        
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}")
        raise

def search_pdf(db, query: str) -> str:
    """Search the stored document and return best matching text."""
    try:
        results = db.max_marginal_relevance_search(
            query=query, 
            k=3, 
            fetch_k=20, 
            lambda_mult=0.5
        )
        
        output = "\n--- Retrieved Context ---\n"
        for i, r in enumerate(results, 1):
            output += f"[Context {i}]\n{r.page_content}\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Failed to search documents: {e}")
        return f"Error retrieving context: {e}"

def initialize_llm_chain(session, db):
    """Initialize the LLM and processing chain."""
    try:
        llm = ChatBedrock(
            model="meta.llama3-70b-instruct-v1:0",
            temperature=0,
            client=session
        )
        
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
You are a government policy advisor. Your task is to explain details in a professional but simple manner, as if assisting a citizen.
Always base your answers strictly on the provided context.
Do not include information from outside the PDF.

When responding:
1) Be polite, helpful, and factual
2) Summarize only what is relevant
3) If context lacks details, say: "I don't have that information available right now."
4) Read the full context before answering
5) Keep answers concise (2–4 sentences)
6) Quote specific details when available

Context: {context}

Question: {question}

Answer:"""
        )
        
        chain = RunnableSequence(
            {
                "question": RunnablePassthrough(),
                "context": lambda q: search_pdf(db, q)
            }
            | prompt
            | llm
        )
        
        logger.info("LLM chain initialized successfully")
        return chain
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM chain: {e}")
        raise

def ask_chatbot(chain, question: str) -> str:
    """Use this function for all government scheme information."""
    try:
        if not question.strip():
            return "Please provide a valid question."
        
        logger.info(f"Processing question: {question[:50]}...")
        response = chain.invoke(question)
        
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info("Question processed successfully")
        return answer
        
    except Exception as e:
        logger.error(f"Failed to process question: {e}")
        return f"I apologize, but I encountered an error: {e}"

def main():
    """Main function to run the chatbot."""
    try:
        # Validate environment
        config = validate_environment()
        logger.info("Environment validation successful")
        
        # Initialize AWS clients
        session, textract_client = initialize_aws_clients(config)
        
        # Load and process documents
        chunks = load_and_process_documents(textract_client, config)
        
        # Initialize vector database
        db = initialize_vector_db(chunks, session, config)
        
        # Initialize LLM chain
        chain = initialize_llm_chain(session, db)
        
        # Start interactive session
        print("\n" + "="*50)
        print("🤖 RAG PDF Chatbot Ready!")
        print("Ask questions about government policies and schemes.")
        print("Type 'exit' to quit.")
        print("="*50)
        
        while True:
            try:
                query = input("\n💬 Ask something (or type 'exit'): ").strip()
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using the chatbot. Goodbye!")
                    break
                
                if not query:
                    continue
                
                answer = ask_chatbot(chain, query)
                print(f"\n🤖 Answer: {answer}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error during conversation: {e}")
                print(f"\n❌ Error: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"❌ Failed to start the chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main
    ()