# AraGenre: Hierarchical Arabic Text Genre Classification

This project was developed as part of the **AraGenre competition**, focusing on the hierarchical classification of Arabic text genres at both broad and specific levels.

We implemented and evaluated two approaches that leverage semantic relationships between Arabic texts and genre definitions:

- **Translation-Based Classification:** Arabic texts and genre definitions are translated into English and analyzed using semantic embeddings and a Large Language Model (LLM) to identify the most relevant genre.
- **Retrieval-Augmented Generation (RAG):** A multilingual embedding model retrieves relevant candidate genres, which are then reranked before the final classification.

The translation-based approach achieved a **Hierarchical F1-score of 0.90**, while the RAG-based approach achieved **0.86**.

This project addresses key challenges in Arabic Natural Language Processing (NLP), including morphological richness, semantic ambiguity, syntactic complexity, and orthographic variation. It also provided practical experience in LLMs, prompt engineering, semantic embeddings, Transformer models, and RAG architectures.

**Technologies & Concepts:** Python · Arabic NLP · Hierarchical Classification · LLMs · RAG · Semantic Embeddings · Transformers · Cross-Encoder Reranking · Prompt Engineering
