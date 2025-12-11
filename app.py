import os
import logging
import traceback
from flask import Flask, request, jsonify, Response
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
from xml.sax.saxutils import escape

# LangChain / AWS imports (same as your original stack)
import boto3
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whatsapp-rag")

# ---------------------------
# Configuration loader
# ---------------------------
def load_config():
    cfg = {
        "AWS_ACCESS_KEY": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": os.getenv("AWS_REGION", "ap-south-1"),
        # S3 path to your PDF (or a local file path like "data/doc.pdf")
        "FILE_PATH": os.getenv("FILE_PATH", "s3://audio-mom/tneb-official-doc.pdf"),
        "PERSIST_DIR": os.getenv("PERSIST_DIR", "db_textract"),
        "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 600)),
        "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 100)),
        # Optional model id: change if you want a different model
        "BEDROCK_MODEL": os.getenv("BEDROCK_MODEL", "meta.llama3-70b-instruct-v1:0"),
        # Embedding model
        "EMBED_MODEL": os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    }

    missing = [k for k, v in cfg.items() if ("AWS" in k or "FILE_PATH" == k) and not v]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return cfg

# ---------------------------
# AWS / Clients initialization (runs once)
# ---------------------------
def init_aws_clients(cfg):
    # Bedrock runtime client
    session = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=cfg["AWS_ACCESS_KEY"],
        aws_secret_access_key=cfg["AWS_SECRET_KEY"],
        region_name=cfg["AWS_REGION"]
    )
    logger.info("Bedrock client initialized")

    # Textract client
    textract_client = boto3.client(
        "textract",
        aws_access_key_id=cfg["AWS_ACCESS_KEY"],
        aws_secret_access_key=cfg["AWS_SECRET_KEY"],
        region_name=cfg["AWS_REGION"]
    )
    logger.info("Textract client initialized")

    return session, textract_client

# ---------------------------
# Load PDF and chunk (runs only if vector DB not present)
# ---------------------------
def load_and_split_pdf(textract_client, cfg):
    logger.info("Loading PDF via Textract: %s", cfg["FILE_PATH"])
    loader = AmazonTextractPDFLoader(cfg["FILE_PATH"], client=textract_client)
    documents = loader.load()
    logger.info("Loaded %d documents", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["CHUNK_SIZE"],
        chunk_overlap=cfg["CHUNK_OVERLAP"]
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks", len(chunks))
    return chunks

# ---------------------------
# Initialize / load vector DB
# ---------------------------
def init_vector_db(chunks, embeddings, cfg):
    persist_dir = cfg["PERSIST_DIR"]
    if os.path.exists(persist_dir):
        logger.info("Loading existing Chroma DB from %s", persist_dir)
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        if not chunks:
            raise RuntimeError("No chunks provided to create DB.")
        logger.info("Creating Chroma DB at %s", persist_dir)
        db = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        logger.info("Vector DB created and persisted")
    return db

# ---------------------------
# Search helper
# ---------------------------
def search_pdf(db, query: str):
    # limit and parameters can be tuned
    results = db.max_marginal_relevance_search(
        query=query, k=3, fetch_k=20, lambda_mult=0.5
    )
    # build context string for LLM prompt
    output = "\n--- Retrieved Context ---\n"
    for r in results:
        output += r.page_content + "\n"
    return output

# ---------------------------
# Build the chain / runnable once
# ---------------------------
def build_chain(session, db, cfg):
    llm = ChatBedrock(
        model=cfg["BEDROCK_MODEL"],
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
1) Be polite, helpful, and factual
2) Summarize only what is relevant
3) If context lacks details, say: “I don’t have that information available right now..”
4) Read the full context before answering.
5) Keep answers concise (2–4 sentences)
6) Quote specific details when available.

Context: {context}

Question: {question}
"""
    )

    # Compose the runnable sequence. The context field is populated by our search function.
    chain = RunnableSequence(
        {
            "question": RunnablePassthrough(),
            "context": lambda q: search_pdf(db, q)
        }
        | prompt
        | llm
    )

    return chain

# ---------------------------
# App bootstrap - runs once at import/start
# ---------------------------
cfg = load_config()
session, textract_client = init_aws_clients(cfg)

# Prepare embeddings and DB - avoid doing textract & embeddings per request
embeddings = BedrockEmbeddings(model_id=cfg["EMBED_MODEL"])

# If DB exists we skip Textract & chunk creation (fast startup)
if os.path.exists(cfg["PERSIST_DIR"]):
    chunks = None
else:
    # This is expensive but only runs once (first start)
    chunks = load_and_split_pdf(textract_client, cfg)

db = init_vector_db(chunks, embeddings, cfg)
chain = build_chain(session, db, cfg)

logger.info("RAG chatbot started and ready")

# ---------------------------
# Flask app & Twilio webhook
# ---------------------------
app = Flask(__name__)

def safe_respond_text(text: str) -> Response:
    """Return a TwiML-safe XML Response for Twilio (escape special xml characters)."""
    resp = MessagingResponse()
    # escape &, <, >, quotes etc
    safe_text = escape(text)
    resp.message(safe_text)
    return Response(str(resp), mimetype="application/xml")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        # Twilio sends Body for SMS/WhatsApp messages
        if "Body" in request.values:
            user_input = request.values.get("Body", "").strip()
            logger.info("Received WhatsApp message: %s", user_input[:200])
            if not user_input:
                return safe_respond_text("I didn't receive any text. Please send your question.")

            # handle simple exit commands
            if user_input.lower() in ["exit", "quit"]:
                return safe_respond_text("Goodbye!")

            # Generate answer from the pre-built chain
            try:
                # invoke the chain; depending on your LLM wrapper you might get .content or plain str
                result = chain.invoke(user_input)
                answer = result.content if hasattr(result, "content") else str(result)
                if not answer:
                    answer = "I don't have that information available right now.."
            except Exception as e:
                logger.error("Error while invoking chain: %s\n%s", e, traceback.format_exc())
                answer = "⚠️ Sorry, an error occurred while generating the answer."

            return safe_respond_text(answer)

        # JSON API fallback
        elif request.is_json:
            data = request.get_json()
            user_input = data.get("input", "").strip()
            if not user_input:
                return jsonify({"error": "No input provided"}), 400
            if user_input.lower() in ["exit", "quit"]:
                return jsonify({"response": "Goodbye!"})
            try:
                result = chain.invoke(user_input)
                answer = result.content if hasattr(result, "content") else str(result)
                if not answer:
                    answer = "I don't have that information available right now.."
            except Exception as e:
                logger.error("Chain error (JSON call): %s\n%s", e, traceback.format_exc())
                answer = "⚠️ Error occurred while processing input."

            return jsonify({"response": answer})

        else:
            return jsonify({"error": "Unsupported request format"}), 400

    except Exception as e:
        logger.exception("Unhandled exception in webhook: %s", e)
        # Twilio expects a proper HTTP response even on error (avoid returning stack traces)
        return safe_respond_text("⚠️ Internal server error. Please try again later.")

if __name__ == "__main__":
    # Listen on 0.0.0.0 so ngrok and external requests can reach it
    app.run(host="0.0.0.0", port=8000, debug=False)
