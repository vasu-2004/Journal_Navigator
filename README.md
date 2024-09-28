# Research Keyword Extraction and Similarity

This project implements a keyword extraction tool designed for research papers, allowing users to extract significant keywords and assess the similarity between different documents. The tool leverages the Groq API and Hugging Face embeddings for advanced keyword extraction and similarity calculations.

# Get Your Api Key
```https://console.groq.com/docs/api-keys```

# Add .env file
```GROQ_API_KEY= <PASTE YOUR API KEY HERE>```

## Features

- Extracts highly relevant keywords from research papers.
- Computes document similarity using various metrics.
- Supports text input from files, direct input, or URLs.

## Requirements

Ensure you have the following dependencies in your `requirements.txt`:

```plaintext
pdfplumber
beautifulsoup4
requests
scikit-learn
chardet
numpy
transformers
torch
langchain_groq
python-dotenv
```
