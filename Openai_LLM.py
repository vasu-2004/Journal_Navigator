import streamlit as st
import subprocess
import sys
import os
import json
import requests
import pdfplumber
import chardet
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import asyncio
import ast
import nltk
import google.generativeai as genai
from openai import OpenAI


api_key = "my_key"
# base_url = "https://api.aimlapi.com/v1"

# client = OpenAI(api_key="", base_url="https://api.aimlapi.com/v1")
nltk.download("punkt")
load_dotenv()


def get_or_create_event_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
DB_FAISS_PATH = "vectorstore/db_faiss"
DB_Path = "openai3/db_faiss"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o")
llm = ChatGroq(model="llama3-8b-8192", temperature=0)
key = os.getenv("GOOGLE_API_KEY")
db = ""
# Check if the vector store already exists
if os.path.exists(DB_Path):
    print("Loading existing FAISS vector store.")
    # Load the embeddings as before
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.load_local(DB_Path, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS vector store.")
    # Load data from the CSV file as before
    loader = CSVLoader(file_path="Final_Research_Dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load_and_split()

    # Create embeddings and vector database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_Path)
# Check if the vector store already exists

    print("Creating new FAISS vector store.")
    # Load data from the CSV file as before
    loader = CSVLoader(
        file_path="Final_Research_Dataset.csv",
        encoding="utf-8",
        csv_args={"delimiter": ","},
    )
    docs = loader.load_and_split()

    # Create embeddings and vector database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))
    db = FAISS(
        embedding_function=OpenAIEmbeddings(),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    db.add_documents(documents=docs)
    db.save_local(DB_Path)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
def extract_token_llama3(text):
    messages = [
        {
            "role": "system",
            "content": (
                "Give a strict comma-separated list of exactly 15 keywords from the following text. "
                "Give a strict comma-separated list of exactly 15 keywords from the following text. "
                "Do not include any bullet points, introductory text, or ending text. "
                "No introductory or ending text strictly"  # Added to ensure can be removed if results deteriorate
                "Do not say anything like 'Here are the keywords.' "
                "Only return the keywords, strictly comma-separated, without any additional words."
            ),
        },
        {"role": "user", "content": text},
    ]

    ai_msg = llm.invoke(messages)
    keywords = ai_msg.content.split("keywords extracted from the text:\n")[-1].strip()
    return keywords.split(",")


def openai_llm(keywords, jif, publisher):
    retriever = db.as_retriever()

    # Set up system prompt
    system_prompt = (
        f"You are a specialized Journal recommender that compares all journals in database to given research paper keywords and based on JIF and publisher gives result."
        f"From the provided context, recommend all journals that are suitable for research paper with {keywords} keywords."
        f"Ensure that you include **every** journal with a Journal Impact Factor (JIF) strictly greater than {jif}, and the Journal must be only from any Publishers in list: {publisher}. And Pls show that jif as in Context database "
        f"Make sure to include both exact matches and related journals, and prioritize including **all relevant high-JIF journals**. "
        f"Present the results in a tabular format( no latex code) with the following columns: Journal Name, Publisher, JIF. "
        f"Ensure no introductory or ending texts are included."
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Ensure the vector dimensions match the FAISS index

    # Invoke the RAG chain
    answer = rag_chain.invoke(
        {"input": f"Keywords: {keywords}, Minimum JIF: {jif},Publisher list: {publisher}"}
    )

    # Inspect the result structure
    return answer["answer"]


def faiss_search(keywords, jif, publisher):
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

    # Load the FAISS index from local storage
    db1 = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # Embed the query
    query_embedding = embeddings.embed_query(keywords)

    # Perform similarity search with FAISS (retrieve top 100 results)
    results = db1.similarity_search_by_vector(query_embedding, k=100)

    # Prepare the context for processing results
    context = "\n\n".join(doc.page_content for doc in results)

    # Minimum JIF value for filtering
    min_jif = jif

    # Check if "no preference" is provided for publishers
    valid_publishers = publisher if publisher != ["no preference"] else None

    # Split the output based on each entry starting with 'Name: '
    entries = re.split(r"\n(?=Name:)", context.strip())

    # Initialize an empty list to hold the dictionaries
    data = []

    # Process each entry
    for entry in entries:
        # Use regex to capture different fields
        name = re.search(r"Name: (.+)", entry)
        jif_match = re.search(r"JIF: (.+)", entry)
        category = re.search(r"Category: (.+)", entry)
        keywords_match = re.search(r"Keywords: (.+)", entry)
        publisher_match = re.search(r"Publisher: (.+)", entry)

        # Filter based on JIF and Publisher
        if jif_match and float(jif_match.group(1)) >= min_jif:
            # If valid publishers are provided, check if the publisher matches
            if (
                valid_publishers is None
                or (publisher_match and publisher_match.group(1) in valid_publishers)
            ):
                try:
                    keywords_list = (
                        json.loads(keywords_match.group(1)) if keywords_match else []
                    )
                except json.JSONDecodeError:
                    keywords_list = []  # Fallback if JSON decoding fails

                data.append(
                    {
                        "Name": name.group(1) if name else None,
                        "JIF": float(jif_match.group(1)) if jif_match else None,
                        "Category": category.group(1) if category else None,
                        "Keywords": keywords_list,
                        "Publisher": publisher_match.group(1) if publisher_match else None,
                    }
                )

    # Sort the data by JIF in decreasing order
    sorted_data = sorted(data, key=lambda x: x["JIF"], reverse=True)

    # Prepare output for markdown table
    if not sorted_data:
        return (
            "| Journal Name | Publisher | JIF |\n"
            "|--------------|-----------|-----|\n"
            "No results found."
        )

    # Create markdown table rows
    table_rows = "\n".join(
        f"| {entry['Name']} | {entry['Publisher']} | {entry['JIF']} |"
        for entry in sorted_data
    )

    # Generate the markdown table header and combine with rows
    output_str = (
        "| Journal Name | Publisher | JIF |\n"
        "|--------------|-----------|-----|\n" + table_rows
    )

    # Prepare the messages for LLaMA
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the best table maker, and you will convert the input into a tabular format with columns: Journal Name, Publisher, and JIF in decreasing order of JIF. "
                "Use a markdown table format. Do not include any introductory or ending text."
            ),
        },
        {"role": "user", "content": output_str},
    ]

    # Invoke the LLaMA model to generate the final output
    ai_msg = llm.invoke(messages)
    return ai_msg.content


# Function to read PDF file content
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text[:6000]


# Function to read text file content with encoding detection
def read_text_file(file):
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result["encoding"]
    return raw_data.decode(encoding)


# Function to clean extracted keywords


def clean_keywords(keywords):
    cleaned_keywords = []
    for keyword in keywords:
        keyword = keyword.strip()
        keyword = re.sub(r"\b\w+@\w+\.\w+\b", "", keyword)  # Remove email addresses
        keyword = re.sub(r"\b\d{10,}\b", "", keyword)  # Remove phone numbers
        if keyword:  # Only add non-empty keywords
            cleaned_keywords.append(keyword)

    # Join the cleaned keywords into a single string, separated by commas
    res = ""
    for i in cleaned_keywords:
        res += i + ","
    return res


# Function to install packages from requirements.txt
def install_requirements():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )


# Streamlit interface
def main():
    st.markdown(
        """
        <h1 style='text-align: center; color: #7b68ee;margin-bottom: -10px;'>JOURNAL FINDER</h1>
        <h3 style='text-align: center; color: #696969;margin-bottom: 30px;'>Extracts The Best Fit for Your Research Paper</h3>"""
        # <hr style='border: 1px solid #FFFFFF; margin: 20px 0;'>  <!-- Horizontal line -->
        ,
        unsafe_allow_html=True,
    )

    # Install requirements
    try:
        install_requirements()
    except Exception as e:
        st.error(f"Error installing requirements: {e}")

    # Sidebar
    st.sidebar.markdown(
        """
        <h5 style='text-align: left; font-weight: bold; color: #6d4ee6;'>RESEARCH PAPER PREFERENCES</h5>
        """,
        unsafe_allow_html=True,
    )

    # Publisher Preferences
    # Function to read and parse the list of publishers from a text file
    def load_publisher_names(file_path):
        with open(file_path, "r") as file:
            # Use ast.literal_eval to safely evaluate the list string
            publishers = ast.literal_eval(file.read())
        return publishers

    # Load publishers from the file
    publisher_options = load_publisher_names("Publishers.txt")
    selected_publishers = st.sidebar.multiselect(
        "Preferred Publishers", publisher_options
    ) or ["no preference"]

    # Impact Factor
    impact_factor = st.sidebar.number_input(
        "Minimum Impact Factor", min_value=0.0, max_value=500.0, step=0.1, format="%.1f"
    )

    # Search Method
    search_method = st.sidebar.selectbox(
        "Choose search method:", ("FAISS Search", "OpenAI LLM")
    )

    # Input for Document
    col1, _ = st.columns([10, 0.1])  # Keep only one column for alignment

    with col1:
        # Label for the text area with custom styling
        st.markdown(
            '<p style="font-weight:bold; margin-bottom: -100px;">GIVE YOUR ABSTRACT HERE</p>',
            unsafe_allow_html=True,
        )

        # Text area placed directly below the label
        document_text1 = st.text_area("", "", height=100)

    # Show Results Button
    if st.button("Show Results"):
        if document_text1:
            with st.spinner("Processing your request..."):  # Spinner starts here
                unique_keywords1 = extract_token_llama3(document_text1)
                cleaned_keywords1 = clean_keywords(unique_keywords1)

                loop = get_or_create_event_loop()
                selected_publishers_str = ", ".join(selected_publishers)

                # Call the selected search method based on dropdown choice
                if search_method == "FAISS Search":
                    results = faiss_search(
                        cleaned_keywords1 + " JIF " + str(impact_factor),
                        impact_factor,
                        selected_publishers,
                    ) 
                elif search_method == "OpenAI LLM":
                    results = openai_llm(
                        cleaned_keywords1, impact_factor, selected_publishers_str
                    )
            # Spinner ends here

            st.markdown(  # Heading is placed outside the spinner block
                "<h3 style='text-align: left; color: #7b68ee;'>RECOMMENDED JOURNALS</h3>",
                unsafe_allow_html=True,
            )

            if search_method == "FAISS Search":
                st.write("FAISS Search Results:")
            elif search_method == "OpenAI LLM":
                st.write("OpenAI LLM Results:")

            st.write(results) # Display the results

        else:
            st.error("Please enter an abstract or upload a document.")
    else:
        st.warning(
            """1. If you have entered the text, click on "Show Results" to get recommendations.\n2. If not, please enter text to get recommendations."""
        )


if __name__ == "__main__":
    main()