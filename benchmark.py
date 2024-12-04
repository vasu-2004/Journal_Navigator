
import subprocess
import sys
import streamlit as st
import os

# Function to install packages from requirements.txt
def install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

# Install requirements
try:
    install_requirements()
except Exception as e:
    st.error(f"Error installing requirements: {e}")

import requests
import pdfplumber
import chardet
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import ast

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(model="llama-3.2-11b-vision-preview", temperature=0)

# Define the functions
def extract_token_llama3(text):
    # Construct the prompt for keyword extraction
    messages = [
        {
            "role": "system",
            "content": (
                "Give a strict comma-separated list of exactly 15 keywords from the following text. "
                "Do not include any bullet points, introductory text, or ending text. "
                "Do not say anything like 'Here are the keywords.' "
                "Only return the keywords, strictly comma-separated, without any additional words."
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg)
    keywords = ai_msg.content.split("keywords extracted from the text:\n")[-1].strip()
    return keywords.split(',')

def llm_similarity(reference_keywords, abstracts_keywords):
    # Construct the prompt for similarity comparison with all abstracts
    prompt = "Calculate similarity percentages between the following reference and abstract keyword lists. Provide only the similarity percentages, formatted with '%' and no additional text or explanation.\n"
    for i, abstract_keywords in enumerate(abstracts_keywords):
        prompt += f"List {i + 1} -> Reference: {', '.join(reference_keywords)}; Abstract {i + 1}: {', '.join(abstract_keywords)}\n"
    
    messages = [
        {
            "role": "system",
            "content": prompt
        }
    ]

    ai_msg = llm.invoke(messages)
    print(ai_msg)
    similarity_results = ai_msg.content.strip()
    st.text(similarity_results)
    return similarity_results

# Streamlit app code
st.title("Abstract Similarity Finder")

# Input fields for abstracts
st.subheader("Enter 5 Abstracts")
abstracts = []
for i in range(5):
    abstract = st.text_area(f"Abstract {i + 1}", key=f"abstract_{i}")
    abstracts.append(abstract)

# Input for reference text
st.subheader("Enter Reference Text")
reference_text = st.text_area("Reference Text", key="reference_text")

# Process on button click
if st.button("Calculate Similarities"):
    if all(abstracts) and reference_text:
        # Extract keywords from the reference text
        st.info("Extracting keywords from the reference text...")
        ref_keywords = extract_token_llama3(reference_text)
        if ref_keywords:
            st.success(f"Keywords for Reference: {', '.join(ref_keywords)}")
        else:
            st.error("Failed to extract keywords from the reference text.")
            ref_keywords = []

        # Extract keywords for all abstracts
        abstracts_keywords = []
        for i, abstract in enumerate(abstracts):
            st.info(f"Processing Abstract {i + 1}...")
            abstract_keywords = extract_token_llama3(abstract)
            if abstract_keywords:
                st.success(f"Keywords for Abstract {i + 1}: {', '.join(abstract_keywords)}")
                abstracts_keywords.append(abstract_keywords)
            else:
                st.error(f"Failed to extract keywords from Abstract {i + 1}.")
                abstracts_keywords.append([])

        # Calculate similarity for all abstracts at once
        if ref_keywords and all(abstracts_keywords):
            st.info("Calculating similarity percentages...")
            similarity_results = llm_similarity(ref_keywords, abstracts_keywords)
            st.write(f"Similarity Results: **{similarity_results}**")
            import pandas as pd


    else:
        st.error("Please fill in all abstract and reference text fields.")
        
# import subprocess
# import sys
# import streamlit as st
# import os
# # Function to install packages from requirements.txt
# def install_requirements():
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

# # Install requirements
# try:
#     install_requirements()
# except Exception as e:
#     st.error(f"Error installing requirements: {e}")

# import requests
# import pdfplumber
# import chardet
# from bs4 import BeautifulSoup
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# import re
# import numpy as np
# from transformers import BertTokenizer, BertModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import CSVLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from dotenv import load_dotenv
# import ast
# load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# # Initialize the LLM
# llm = ChatGroq(model="llama-3.2-11b-vision-preview", temperature=0)

# # Define the functions
# def extract_token_llama3(text):
#     # Construct the prompt for keyword extraction
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "Give a strict comma-separated list of exactly 15 keywords from the following text. "
#                 "Do not include any bullet points, introductory text, or ending text. "
#                 "No introductory or ending text strictly" # Added to ensure can be removed if results deteriorate
#                 "Do not say anything like 'Here are the keywords.' "
#                 "Only return the keywords, strictly comma-separated, without any additional words."
                
#             )
#         },
#         {
#             "role": "user",
#             "content": text
#         }
#     ]
#     ai_msg = llm.invoke(messages)
#     print(ai_msg)
#     keywords = ai_msg.content.split("keywords extracted from the text:\n")[-1].strip()
#     return keywords.split(',')
    

# def llm_similarity(list1, list2):
#     # Construct the prompt for similarity comparison
#     messages = [
#         (
#             "system",
#             f"Calculate similarity percentage between these two lists of keywords by considering semantic relevance and context. List 1 -> {', '.join(list1)}; List 2 -> {', '.join(list2)}. Provide only the similarity percentage, with no additional text or explanation."
#         ) # Placeholder for the second part of the input, not needed here
#     ]
    
#     ai_msg = llm.invoke(messages)
#     # Output the similarity percentage
#     similarity_percentage = ai_msg.content.strip()  # Clean any leading/trailing whitespace
#     print(similarity_percentage)
#     st.text(similarity_percentage)
#     return similarity_percentage

# # Streamlit app code
# st.title("Abstract Similarity Finder")

# # Input fields for abstracts
# st.subheader("Enter 5 Abstracts")
# abstracts = []
# for i in range(5):
#     abstract = st.text_area(f"Abstract {i + 1}", key=f"abstract_{i}")
#     abstracts.append(abstract)

# # Input for reference text
# st.subheader("Enter Reference Text")
# reference_text = st.text_area("Reference Text", key="reference_text")

# # Process on button click
# if st.button("Calculate Similarities"):
#     if all(abstracts) and reference_text:
#         # Extract keywords from the reference text
#         st.info("Extracting keywords from the reference text...")
#         ref_keywords = extract_token_llama3(reference_text)
#         if ref_keywords:
#             st.success(f"Keywords for Reference: {', '.join(ref_keywords)}")
#         else:
#             st.error("Failed to extract keywords from the reference text.")
#             ref_keywords = []

#         # Process each abstract and display similarity results
#         for i, abstract in enumerate(abstracts):
#             st.info(f"Processing Abstract {i + 1}...")
#             abstract_keywords = extract_token_llama3(abstract)
#             if abstract_keywords:
#                 st.success(f"Keywords for Abstract {i + 1}: {', '.join(abstract_keywords)}")
#                 similarity = llm_similarity(abstract_keywords, ref_keywords)
#                 st.write(f"Similarity between Abstract {i + 1} and Reference: **{similarity}**")
#             else:
#                 st.error(f"Failed to extract keywords from Abstract {i + 1}.")
#     else:
#         st.error("Please fill in all abstract and reference text fields.")