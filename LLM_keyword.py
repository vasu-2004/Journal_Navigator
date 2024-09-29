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
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = 'vectorstore/db_faiss'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
llm = ChatGroq(model="llama3-8b-8192",temperature=0)
db=""
# Check if the vector store already exists
# Check if the vector store already exists
if os.path.exists(DB_FAISS_PATH):
    print("Loading existing FAISS vector store.")
    # Load the embeddings as before
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS vector store.")
    # Load data from the CSV file as before
    loader = CSVLoader(file_path="Final_Research_Dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    
    # Create embeddings and vector database
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)


chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
def extract_token_llama3(text):
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
    
def llm_similarity(list1, list2):
    # Create the messages with formatted strings for list1 and list2
    messages = [
        (
            "system",
            f"Compare the similarity between the two lists of tokens provided where 1st list is of a research paper and the 2nd text is about the website that receives that research paper so on a scale of 0 to 100 how much is that research paper suitable for the website: List 1 -> {', '.join(list1)}; List 2 -> {', '.join(list2)}. Provide only the similarity percentage as an integer, without any additional text, explanations, or special symbols."
        ) # Placeholder for the second part of the input, not needed here
    ]
    
    # Invoke the LLM with the prepared messages
    ai_msg = llm.invoke(messages)
    
    # Output the similarity percentage
    similarity_percentage = ai_msg.content.strip()  # Clean any leading/trailing whitespace
    print(similarity_percentage)
    st.text(similarity_percentage)
    return similarity_percentage

def final_res(list1,index_f,ts):
    # Ensure that list1 is a list and handle the case where it's empty
    
    
    query = f"You are a research paper journal recommender system. Given the following list of keywords, recommend a suitable journal for this research paper with their similarity percentage the preferred published timeline is {ts} and preferred impact factor is {index_f}. Keywords: {list1}" 
    result = chain({"question": query, "chat_history": st.session_state.get('history', [])})
   
    return (result['answer'])
    

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
    encoding = result['encoding']
    return raw_data.decode(encoding)

# Function to scrape text from a URL
def scrape_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join(p.get_text() for p in paragraphs)
        return text
    except Exception as e:
        st.write(f"Error scraping the URL: {e}")
        return ""

# Function to extract keywords using TF-IDF


# Function to clean extracted keywords


def clean_keywords(keywords):
    cleaned_keywords = []
    for keyword in keywords:
        keyword = keyword.strip()
        keyword = re.sub(r'\b\w+@\w+\.\w+\b', '', keyword)  # Remove email addresses
        keyword = re.sub(r'\b\d{10,}\b', '', keyword)  # Remove phone numbers
        if keyword:  # Only add non-empty keywords
            cleaned_keywords.append(keyword)
    
    # Join the cleaned keywords into a single string, separated by commas
    return ', '.join(cleaned_keywords)


# Function to calculate cosine similarity




# Function to calculate Euclidean distance (as a similarity measure)




# Function to calculate pairwise cosine similarity between words
 # Return average similarity percentage

# Function to calculate pairwise Jaccard similarity between words

    similarities = []
    for word1 in list1:
        max_similarity = 0
        for word2 in list2:
            set1, set2 = set(word1), set(word2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jaccard_sim = intersection / union if union > 0 else 0
            max_similarity = max(max_similarity, jaccard_sim)
        similarities.append(max_similarity)
    
    return np.mean(similarities) * 100  # Return average similarity percentage


# Streamlit interface
st.title("Document Keyword Extractor and Similarity Checker")
st.write("Extract top 10 unique keywords from two documents and compute the similarity between them.")

# Input for Document 1
input_option1 = st.radio("Choose input method for Document 1:", ("Upload a file", "Enter text directly", "Provide a URL"))
document_text1 = ""

if input_option1 == "Upload a file":
    uploaded_file1 = st.file_uploader("Choose a document", type=["txt", "pdf"], key="doc1")
    if uploaded_file1 is not None:
        try:
            if uploaded_file1.type == "application/pdf":
                document_text1 = read_pdf(uploaded_file1)
            else:
                document_text1 = read_text_file(uploaded_file1)
            st.subheader("Document 1 Content:")
            st.write(document_text1[:10000] + "...")  # Display first 10000 characters
        except Exception as e:
            st.write(f"Error reading the file: {e}")
elif input_option1 == "Enter text directly":
    document_text1 = st.text_area("Enter the text for Document 1:", "")
    if document_text1:
        st.subheader("Entered Text Content for Document 1:")
        st.write(document_text1[:500] + "...")  # Display first 500 characters
else:  # URL option
    url1 = st.text_input("Enter the URL for Document 1:")
    if url1:
        document_text1 = scrape_url(url1)
        if document_text1:
            st.subheader("Scraped Content for Document 1:")
            st.write(document_text1[:10000] + "...")  # Display first 10000 characters

# Preferences input after Document 1
st.write("---")
st.subheader("Research Paper Preferences")
publisher_options = ["IEEE", "Springer", "Elsevier", "ACM", "Wiley"]
selected_publishers = st.multiselect("Select preferred publishers:", publisher_options)

# Impact Factor
# Initialize impact factor variable
impact_factor = 0.0  # Default value

# Create a slider for impact factor
impact_factor = st.slider(
    "Select Minimum Impact Factor",
    min_value=0.0,
    max_value=500.0,
    step=0.1,
    value=impact_factor,  # Set initial value from the variable
)

# Display the current impact factor
st.write("Current Impact Factor:", impact_factor)

# Changed timeline to fixed intervals (6 months, 1 year, etc.)
timeline_options = ["6 months", "1 year", "1.5 years", "2 years", "2.5 years", "3 years"]
timeline_selection = st.selectbox("Timeline for publication:", timeline_options)
# Checkbox to allow users to ignore preferences
ignore_preferences = st.checkbox("Ignore impact factor and timeline preferences", value=False)
# Input for Document 2


 # Display first 10000 characters

# Process and extract keywords if both document texts are available
if document_text1 :
    # Use TF-IDF for keyword extraction
    unique_keywords1 = extract_token_llama3(document_text1)
    
    
    st.write("---")
    st.subheader("Extracted Keywords:")
    st.write("**Document 1 Keywords:**")
    st.markdown("\n".join(f"- {keyword}" for keyword in unique_keywords1))
    
    
    # Clean keywords
    cleaned_keywords1 = clean_keywords(unique_keywords1)
    
    # Calculate similarities
    
    # Display similarity percentages
    st.write("### SIMILARITY MEASURES:")

    def format_similarity_text(label, percentage):
        color = 'red' if percentage < 50 else 'lightgreen'
        return f"<div style='font-size: 20px; color: white; font-weight: bold;'>{label}: <span style='color:{color}; font-weight:bold; font-size:24px;'>{percentage:.2f}%</span></div>"


    # Display user preferences
    
    if ignore_preferences or (impact_factor > 0 and timeline_selection):
        # Call the final_res function only if preferences are valid
        st.markdown(final_res(cleaned_keywords1, impact_factor, timeline_selection))
    else:
        st.warning("Please set your preferences or choose to ignore them before proceeding.")