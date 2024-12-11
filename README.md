# rag-research-assistant
Project Overview

This project involves the development of a Research Assistant Tool that utilizes a Retrieval-Augmented Generation (RAG) approach to provide precise and contextually relevant answers to queries based on a dataset of scientific papers. By combining advanced retrieval techniques and large language models (LLMs), this system delivers accurate and concise responses for research purposes.

Key Features:
	1.	Semantic Search: Implemented using FAISS (Facebook AI Similarity Search) for efficient and scalable similarity-based retrieval.
	2.	LLM Integration: Uses cutting-edge language models for response generation.
	3.	Fine-tuning Capabilities: Enhances performance by fine-tuning LLMs with a domain-specific query-response dataset.
 
 Project Workflow

  1. Dataset Preparation:
    *  Preprocess and clean the dataset of scientific papers.
    *  Convert text data into embeddings using a pre-trained embedding model.
	2. Context Retrieval:
	  *  Implement FAISS for efficient similarity search.
	  *  Retrieve the most relevant contexts from the dataset based on user queries.
	3. Response Generation:
	  *  Use a base LLM to process the retrieved context and generate human-like responses.
	  *  Fine-tune the model for improved accuracy and domain-specific performance.
	4. Fine-tuning:
	  *  Train the Qwen-1.5b model on a custom query-response dataset to enhance its performance for scientific research queries.

Technology Stack

Retrieval System:

  *  FAISS (Facebook AI Similarity Search):
  *  Used for indexing and retrieving similar documents efficiently.
	*  Handles high-dimensional embeddings for semantic similarity.

Language Models (LLMs):
  
  1. Qwen-1.5B (Model):
	  *  Purpose: Used as the primary model for initial experimentation.
	2. Mistral-7B:
	  *  A highly efficient LLM with 7 billion parameters.
	  *  Known for its superior performance in understanding and generating text.
	3. Fine-Tuned Qwen-1.5B:
	  *  A domain-adapted version of the Qwen-1.5B model.
	  *  Fine-tuned on a custom dataset of scientific paper queries and responses.
To run the system, run all blocks of code in the NLP_RAG_Research_Assistant notebook. This will generate a Gradio UI and you will be able to interact with the LLM
