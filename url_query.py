"""
This is a RAG application using llama2 which is built on langchain. It does the followings:
1. Users enters the URL(s)
2. Users asks a question
3. The application returns the answer from the provided URL(s)
"""

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

#Process the Url
def input_process(urls, question):
    """
    The method process the users input. The users input takes two arguments viz. URL(s) and the users question.
    """
    model = Ollama(model='llama2')

    #Load the Url's from web

    url_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in url_list]
    docs_list = [item for sublist in docs for item in sublist]

    #Break the document into chunks

    text_split = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_split = text_split.split_documents(docs_list)

    #Convert into embeddings and store in a vector db

    vector_store = Chroma.from_documents(
        documents = doc_split,
        collection_name = "rag-chroma",
        embedding= OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vector_store.as_retriever()

    #Perform RAG

    after_rag_template = """answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

#Use Streamlit

st.title("RAG Application using llama2")
st.write("Enter URLs and ask a question")

#Input Fields
urls = st.text_area("Enter URLs separated by lines")
question = st.text_input("Question")

if st.button('Search the URL'):
    answer = input_process(urls, question)
    st.text_area("Answer", value = answer, height=500)
