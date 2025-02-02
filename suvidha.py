import os
import re
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import speech_recognition as sr
import pyttsx3
from google.cloud import vision
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def initialize_gemini_chat():
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        return chat
    except Exception:
        return None

def initialize_models():
    try:
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
        return gpt2_model, gpt2_tokenizer
    except Exception:
        return None, None

def get_pdf_text(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error reading PDF: {str(e)}"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(text_chunks)]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, just say, "I don't have enough information to answer that question." Please don't provide incorrect information.
    
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def hybrid_search(vector_store, user_question):
    docs = vector_store.similarity_search(user_question, k=4)
    tokenized_docs = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(user_question.split())
    ranked_docs = [docs[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
    return ranked_docs[:5]

def process_document_query(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = hybrid_search(vector_store, user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response.get("output_text", "No response generated.")
    except Exception as e:
        return f"Error processing query: {str(e)}"

def classify_intent(user_input):
    if re.search(r"document|pdf|file|text|read|extract|analyze", user_input, re.IGNORECASE):
        return "document"
    return "general"


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return redirect(request.url)
    document = request.files['document']
    filename = secure_filename(document.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    document.save(file_path)
    raw_text = get_pdf_text(file_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return jsonify({"message": "Document processed successfully!"})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('user_input')
    intent = classify_intent(user_input)
    chat_instance = initialize_gemini_chat()
    if intent == "document":
        response = process_document_query(user_input)
    else:
        response = chat_instance.send_message(user_input, stream=False).text.strip() if chat_instance else "I'm having trouble connecting to my knowledge base. Please try again later."
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT from Render, default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
