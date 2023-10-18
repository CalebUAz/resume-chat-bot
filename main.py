import os
import sys

import dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PIL import Image


# Monkey patch sqlite3 to use pysqlite3
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# Load API key from .env file
script_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(script_directory, "utils/data")
dotenv.load_dotenv(os.path.join(script_directory, ".env"))
API_KEY = os.getenv("OPENAI_API_KEY")
NAME = os.getenv("NAME") or "John Doe"
if not API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")


def init_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def init_vector_store():
    # load resume
    loader = PyPDFLoader(os.path.join(data_directory, "resume.pdf"))
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    resume_pages = loader.load_and_split(text_splitter=text_splitter)

    vector_store = Chroma.from_documents(
        documents=resume_pages, embedding=OpenAIEmbeddings()
    )
    return vector_store


def init_qa_chain(vector_store, llm):
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")


# Initialize vector store and LLM
vector_store = init_vector_store()
llm = init_llm()

# Initialize retriever and QA chain
qa_chain = init_qa_chain(vector_store, llm)

# Streamlit app
st.set_page_config(
    page_title=f"{NAME}GPT",
    page_icon=":robot_face:",
    layout="centered",
)


st.title(f"{NAME}'s ResumeGPT")

resume_image = Image.open(os.path.join(data_directory, "Resume_1.tiff"))
st.sidebar.image(resume_image, use_column_width="auto", caption=f"{NAME}'s Resume")

with open(os.path.join(data_directory, "resume.pdf"), "rb") as f:
    st.sidebar.download_button(
        "Download Resume",
        f,
        file_name=f"{NAME}_resume.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input(f"Ask me a question about {NAME}'s resume"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Loading..."):
        response = qa_chain({"query": prompt})["result"]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})