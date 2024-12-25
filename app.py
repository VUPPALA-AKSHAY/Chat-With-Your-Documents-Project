
!pip install langchain-community
!pip install pypdf
!pip install docx2txt
!pip install fastembed
!pip install faiss-cpu
!pip install groq
!pip install langchain-groq
!pip install pythonenv
!pip install --upgrade gradio


from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
import os

loader = PyPDFDirectoryLoader('/content/DOCUMENTDATA')
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
text = text_splitter.split_documents(data)


embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db = FAISS.from_documents(text, embeddings)


retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

query = "what is nlp"
retrieved_docs = retriever.invoke(query)
retrieved_docs


with open('.env', 'w') as f:
    f.write('GROQ_API_KEY=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')

load_dotenv()

client = Groq(api_key="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")


client = Groq(api_key=os.getenv('GROQ_API_KEY'))

from groq import Groq
client = Groq(api_key="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")


llm = ChatGroq(api_key="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", model_name="llama3-70b-8192")

import os
os.environ["GROQ_API_KEY"] = "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"
print(os.getenv("GROQ_API_KEY"))

print(os.getenv("GROQ_API_KEY"))



llm = ChatGroq(model_name="llama3-70b-8192")

OUTPT = "What is Network"
print(llm.invoke(OUTPT))


prompt_template = """
greet the user

Context: {context}

Question: {question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={'prompt': prompt},
    return_source_documents=True
)


memory_1 = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

qa_1 = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    memory=memory_1
)

query = "what is nlp"
result = qa_1.invoke(query)
result
result['answer']


def _answer(query):
    result = qa_1.invoke(query)
    return result['answer']


iface = gr.Interface(
    fn=_answer,
    inputs=gr.Textbox(label="Ask a question "),
    outputs=gr.Textbox(label="Answer"),
    live=True,
)
iface.launch(share=True)
