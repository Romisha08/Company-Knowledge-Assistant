# InsureLLM

A Retrieval-Augmented Generation (RAG) application
for answering questions using insurance-related documents.

## Features

- Document ingestion
- Text chunking
- Vector database creation
- Semantic search
- LLM-based question answering
- Answer evaluation

## Tech Stack

- Python
- LangChain
- ChromaDB
- OpenAI / Hugging Face
- Gradio

## Project Structure

insureLLM/
├── evaluation/
├── implementation/
├── knowledge-base/
├── app.py
├── evaluator.py
├── requirements.txt
├── .env.example
└── README.md

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project:

cd insureLLM

Create a virtual environment:

python -m venv .venv

Activate the environment:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Environment Variables

Create a `.env` file and add your API keys.

Refer to `.env.example` for the required variables.

## Usage

Run the ingestion pipeline:

python implementation/ingest.py

Run the application:

python app.py

Run evaluation:

python evaluator.py