import os 
import torch
import pipeline
# import importlib
# from langchain_ollama.chat_models import ChatOllama
# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
load_dotenv(find_dotenv())

def load_model():
     llm = OllamaLLM(
          model= os.getenv("phi"),
          base_url= os.getenv("http://localhost:11434"),
          temperature=0.9,
          max_tokens=512,
     )
     return llm

def load_documents(file_path):
     if not os.path.exists(file_path):
          raise FileNotFoundError(f"File {file_path} not found.")
     try:
          if file_path.lower().endswidth(".pdf"):
               loader = PyPDFLoader(file_path)
          elif file_path.lower().endswidth(".txt"):
               loader = TextLoader(file_path)
          elif file_path.lower().enswidth(".docx"):
               loader = Docx2txtLoader(file_path)
          else:
               raise ValueError("Unsupported File Type for {file_path}. Use .pdf or .txt or .docx.")
          documents = loader.load()
          return documents
     except Exception as e:
          raise Exception(f"Error loading {file_path}: {str(e)}")

def chunk_documents(documents):
     text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=500,
          chunk_overlap=100
     )
     chunks = text_splitter.split_documents(documents)
  
     return chunks
     
#run once for downloading embedding model (internet required)
# from sentence_transformers import SentenceTransformer
# model_name = "all-MiniLM-L6-v2"
# save_path = "./models/all-MiniLM-L6-v2"
# model = SentenceTransformer(model_name)
# model.save(save_path)

def create_vector_store(chunks):
     embeddings = HuggingFaceEmbeddings(
          model_name="./models/all-MiniLM-L6-v2",
          model_kwargs={"local_files_only": True}
     )
     vector_store = FAISS.from_documents(chunks, embeddings)
     return vector_store

def prompt_template():
     template = """Use the following context to answer the question. Only use information from the context, and do not make up answer:
     {context}
     Question: {question}
     Answer: """ 
     return PromptTemplate(template=template, input_variables=["context", "question"])

def create_rag_chain(vector_store, llm):
     rag_chain = RetrievalQA.from_chain_type(
          llm=llm,
          chain_type="stuff",
          retriever=vector_store.as_retriever(search_kwargs={"k":2}),
          return_source_documents=True,
          chain_type_kwargs={"prompt": prompt_template()}
     )
     return rag_chain

def answer_query(rag_chain, query):
     result = rag_chain({"query": query})
     answer = result["result"]
     sources = result["source_documents"]
     return answer, sources

def command_line_interface():
     llm = load_model()
     file_path = input("Enter the path to your files: ")

     try:
          documents = load_documents(file_path)
          chunks = chunk_documents(documents)
          vector_store = create_vector_store(chunks)
          rag_chain = create_rag_chain(vector_store, llm)

          print("\n Chatbot is ready to answer your questions!")
          while True:
               query = input("\nAsk a question (or type 'exit' to quit): ")
               if query.lower() == "exit":
                    break
               try:
                    answer, sources = answer_query(rag_chain, query)
                    print("\nAnswer:", answer)
                    print("\nSources:")
                    for i, doc in enumerate(sources, 1):
                         print(f"Source {i}: {doc.page_content[:200]}...")
               except Exception as e:
                    print(f"Error processing query: {e}")
     except Exception as e:
          print(f"Error processing file: {e}")

def streamlit_interface():
     st.title("Local RAG Chatbot (Powered by Ollama)")
     st.write("Upload a text or pdf or docx file and ask questions based on its content.")

     uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
     if uploaded_file:
          file_extension = os.path.splitext(uploaded_file.name)[1].lower()
          file_path = f"temp_file{file_extension}"

          try:
               with open(file_path, "wb") as f:
                     content = uploaded_file.read()
                     st.write(f"Saving file: {uploaded_file.name}, Size: {len(content)} bytes")
                     f.write(content)
               
               if not os.path.exists(file_path):
                    st.error(f"Failed to save temporary file at {file_path}")
                    return
               
               try:
                    llm = load_model()
                    documents = load_documents(file_path)
                    chunks = chunk_documents(documents)
                    vector_store = create_vector_store(chunks)
                    rag_chain = create_rag_chain(vector_store , llm)

                    query = st.text_input("Ask a question: ")
                    if query:
                         try:
                              with st.spinner("Processing query..."):
                                   answer, sources = answer_query(rag_chain, query)
                              st.write("**ANSWER:**", answer)
                              st.write("**SOURCES:**")
                              for i, doc in enumerate(sources, 1):
                                   st.write(f"Source {i}: {doc.page_content[:200]}...")
                         except Exception as e:
                              st.error(f"Error processing query: {e}")
               except Exception as e:
                    st.error(f"Error processing file: {e}")
               finally:
                    if os.path.exists(file_path):
                         try:
                              os.remove(file_path)
                         except Exception as e:
                              st.warning(f"Failed to delete temporary file: {e}")
          except Exception as e:
               st.error(f"Error saving file: {e}")

if __name__ == "__main__":
     streamlit_interface()

