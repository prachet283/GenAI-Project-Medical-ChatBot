from flask import Flask, render_template, jsonify, request,session
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name = index_name
)

retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    #repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    chat_history = session.get("chat_history", [])
    input = msg
    print(input)
    #response = rag_chain.invoke({"input": msg})
    response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
    # Append user + bot messages to history
    chat_history.append(("human", msg))
    chat_history.append(("ai", response["answer"]))
    # Save back to session
    session["chat_history"] = chat_history
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000,debug=True,use_reloader=False)