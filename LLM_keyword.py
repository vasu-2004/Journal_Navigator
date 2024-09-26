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
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
llm = ChatGroq(model="llama3-8b-8192",temperature=0)

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

    
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculate_bert_similarity(text1, text2):
    embedding1 = get_bert_embeddings(text1)
    embedding2 = get_bert_embeddings(text2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0] * 100  # Return percentage
    return similarity

# Function to read PDF file content
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

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
def extract_keywords_tfidf(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tfidf_matrix.toarray().flatten().argsort()[::-1]
    top_keywords = feature_array[tfidf_sorting][:top_n]
    return top_keywords.tolist()

# Function to clean extracted keywords
def clean_keywords(keywords):
    cleaned_keywords = []
    for keyword in keywords:
        keyword = keyword.strip()
        keyword = re.sub(r'\b\w+@\w+\.\w+\b', '', keyword)  # Remove email addresses
        keyword = re.sub(r'\b\d{10,}\b', '', keyword)  # Remove phone numbers
        if keyword:  # Only add non-empty keywords
            cleaned_keywords.append(keyword)
    return cleaned_keywords

# Function to calculate cosine similarity
def calculate_cosine_similarity(list1, list2):
    vectorizer = TfidfVectorizer().fit_transform([' '.join(list1), ' '.join(list2)])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1] * 100  # Return percentage

# Function to calculate Jaccard similarity
def calculate_jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union * 100 if union > 0 else 0

# Function to calculate Overlap Coefficient
def calculate_overlap_coefficient(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    return intersection / min(len(set1), len(set2)) * 100 if min(len(set1), len(set2)) > 0 else 0

# Function to calculate Euclidean distance (as a similarity measure)
def calculate_euclidean_similarity(list1, list2):
    vectorizer = TfidfVectorizer().fit_transform([' '.join(list1), ' '.join(list2)])
    vectors = vectorizer.toarray()
    euclidean_distance = euclidean_distances(vectors)[0][1]
    max_distance = np.sqrt(len(list1) + len(list2))  # Max possible distance
    similarity = (1 - (euclidean_distance / max_distance)) * 100
    return similarity

# Function to calculate Dice Coefficient
def calculate_dice_coefficient(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    return 2 * intersection / (len(set1) + len(set2)) * 100 if (len(set1) + len(set2)) > 0 else 0

# Function to calculate pairwise cosine similarity between words
def calculate_word_cosine_similarity(list1, list2):
    vectorizer = TfidfVectorizer().fit_transform(list1 + list2)
    vectors = vectorizer.toarray()
    similarities = []
    
    for word1_vector in vectors[:len(list1)]:
        max_similarity = 0
        for word2_vector in vectors[len(list1):]:
            similarity = cosine_similarity([word1_vector], [word2_vector])[0][0]
            max_similarity = max(max_similarity, similarity)  # Taking max similarity for each word
        similarities.append(max_similarity)
    
    return np.mean(similarities) * 100  # Return average similarity percentage

# Function to calculate pairwise Jaccard similarity between words
def calculate_word_jaccard_similarity(list1, list2):
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
h_index = st.number_input("Minimum H-index:", min_value=0, value=0)
timeline_years = st.number_input("Timeline for publication (years):", min_value=0, value=1)

# Input for Document 2
st.write("---")
input_option2 = st.radio("Choose input method for Document 2:", ("Upload a file", "Enter text directly", "Provide a URL"))
document_text2 = ""

if input_option2 == "Upload a file":
    uploaded_file2 = st.file_uploader("Choose a second document", type=["txt", "pdf"], key="doc2")
    if uploaded_file2 is not None:
        try:
            if uploaded_file2.type == "application/pdf":
                document_text2 = read_pdf(uploaded_file2)
            else:
                document_text2 = read_text_file(uploaded_file2)
            st.subheader("Document 2 Content:")
            st.write(document_text2[:10000] + "...")  # Display first 10000 characters
        except Exception as e:
            st.write(f"Error reading the file: {e}")
elif input_option2 == "Enter text directly":
    document_text2 = st.text_area("Enter the text for Document 2:", "")
    if document_text2:
        st.subheader("Entered Text Content for Document 2:")
        st.write(document_text2[:500] + "...")  # Display first 500 characters
else:  # URL option
    url2 = st.text_input("Enter the URL for Document 2:")
    if url2:
        document_text2 = scrape_url(url2)
        if document_text2:
            st.subheader("Scraped Content for Document 2:")
            st.write(document_text2[:10000] + "...")  # Display first 10000 characters

# Process and extract keywords if both document texts are available
if document_text1 and document_text2:
    # Use TF-IDF for keyword extraction
    unique_keywords1 = extract_token_llama3(document_text1)
    unique_keywords2 = extract_token_llama3(document_text2)
    
    st.write("---")
    st.subheader("Extracted Keywords:")
    st.write("**Document 1 Keywords:**")
    st.markdown("\n".join(f"- {keyword}" for keyword in unique_keywords1))
    st.write("**Document 2 Keywords:**")
    st.markdown("\n".join(f"- {keyword}" for keyword in unique_keywords2))

    
    # Clean keywords
    cleaned_keywords1 = clean_keywords(unique_keywords1)
    cleaned_keywords2 = clean_keywords(unique_keywords2)
    
    # Calculate similarities
    cosine_similarity_percentage = calculate_cosine_similarity(cleaned_keywords1, cleaned_keywords2)
    jaccard_similarity_percentage = calculate_jaccard_similarity(cleaned_keywords1, cleaned_keywords2)
    overlap_coefficient_percentage = calculate_overlap_coefficient(cleaned_keywords1, cleaned_keywords2)
    euclidean_similarity_percentage = calculate_euclidean_similarity(cleaned_keywords1, cleaned_keywords2)
    dice_coefficient_percentage = calculate_dice_coefficient(cleaned_keywords1, cleaned_keywords2)
    calculate_word_jaccard_percentage=calculate_word_jaccard_similarity(cleaned_keywords1, cleaned_keywords2)
    calculate_word_cosine_percentage=calculate_word_cosine_similarity(cleaned_keywords1, cleaned_keywords2)
    bert_similarity_percentage = calculate_bert_similarity(document_text1, document_text2)
    llm_sim_percentage=eval(llm_similarity(cleaned_keywords1,cleaned_keywords2))
    # Display similarity percentages
    st.write("### SIMILARITY MEASURES:")

    def format_similarity_text(label, percentage):
        color = 'red' if percentage < 50 else 'lightgreen'
        return f"<div style='font-size: 20px; color: white; font-weight: bold;'>{label}: <span style='color:{color}; font-weight:bold; font-size:24px;'>{percentage:.2f}%</span></div>"

    
    st.markdown(format_similarity_text("COSINE SIMILARITY", cosine_similarity_percentage), unsafe_allow_html=True)
    st.markdown(format_similarity_text("JACCARD SIMILARITY", jaccard_similarity_percentage), unsafe_allow_html=True)
    st.markdown(format_similarity_text("OVERLAP COEFFICIENT", overlap_coefficient_percentage), unsafe_allow_html=True)
    st.markdown(format_similarity_text("EUCLIDEAN SIMILARITY", euclidean_similarity_percentage), unsafe_allow_html=True)
    st.markdown(format_similarity_text("DICE COEFFICIENT", dice_coefficient_percentage), unsafe_allow_html=True)
    st.markdown(format_similarity_text("WORD COSINE SIMILARITY",calculate_word_cosine_percentage ), unsafe_allow_html=True)
    st.markdown(format_similarity_text("WORD JACCARD SIMILARITY", calculate_word_jaccard_percentage), unsafe_allow_html=True)
    st.markdown(format_similarity_text("BERT SIMILARITY", bert_similarity_percentage), unsafe_allow_html=True)
    st.markdown(format_similarity_text("LLM SIMILARITY", llm_sim_percentage), unsafe_allow_html=True)
    # Display user preferences
    st.write("---")
    st.subheader("Your Preferences:")
    st.write(f"Preferred Publishers: {', '.join(selected_publishers) if selected_publishers else 'None'}")
    st.write(f"Minimum H-index: {h_index}")
    st.write(f"Timeline for publication: {timeline_years} years")