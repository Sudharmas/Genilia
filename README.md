Genilia: Autonomous Multi-Agent GenAI Ecosystem

Technical Documentation & Architecture Report

Date: November 2025
Version: 1.0.0
Author: [Your Name]

1. Executive Summary

Genilia is an enterprise-grade, local-first Generative AI assistant designed to solve complex customer support queries. Unlike traditional chatbots that rely on a single model, Genilia utilizes a multi-agent microservices architecture. It intelligently orchestrates tasks between three specialized workers: a Retrieval-Augmented Generation (RAG) agent for internal policy retrieval, an Action Agent for live web lookups, and a Fine-Tuned Small Language Model (SLM) for high-precision technical specifications.

The system is designed for privacy, cost-efficiency, and resilience, operating 100% locally using quantized open-source models (Phi-3, Qwen2-0.5B) via Ollama and Hugging Face, removing dependency on expensive cloud APIs while maintaining high performance.

2. System Architecture

The system follows a Hub-and-Spoke architecture pattern orchestrated by a central Mission Control Plane (MCP).

2.1 Core Components

Mission Control Plane (MCP Server - Port 8002):

Role: The orchestrator and "brain" of the system.

Tech: FastAPI, LangChain.

Key Functions:

Session Management: Maintains in-memory chat history for context awareness.

Query Condensing: Uses an LLM to rewrite follow-up questions (e.g., "How much does it cost?") into standalone queries based on history.

Intelligent Routing: Classifies user intent to delegate tasks to the correct worker agent.

Resilience: Implements fallback logic (if RAG fails, try Action Agent).

RAG Agent (Port 8000):

Role: The "Librarian" for static internal knowledge.

Tech: ChromaDB (Vector Store), HuggingFace Embeddings (all-MiniLM-L6-v2).

Features:

Metadata-aware filtering (searches specific product lines vs. general docs).

Dedicated summarization and Q&A chains.

Stateless database connection architecture to prevent locking.

Action Agent (Port 8001):

Role: The "Researcher" for live, external data.

Tech: LangGraph, Tavily API.

Features:

Equipped with custom tools to perform site-restricted searches (e.g., ensuring answers come only from verified company domains).

Finetune Agent (Port 8003):

Role: The "Specialist" for deep technical specs.

Tech: PyTorch, PEFT/LoRA, Qwen2-0.5B-Instruct.

Features:

Runs a custom SLM trained specifically on the organization's data using 4-bit quantization for efficiency.

3. Key Features & Capabilities

3.1 Hybrid "Router" Architecture

Instead of forcing one model to do everything, Genilia routes queries:

"What is the return policy?" → RAG Agent (Retrieves PDF policy).

"What are the exact dimensions of the X-1000?" → Finetune Agent (Uses trained weights for precision).

"What's new on the blog?" → Action Agent (Live web search).

3.2 Automated Fine-Tuning Pipeline (MLOps)

We implemented a complete offline ML pipeline to create the specialist agent:

Data Generation: A script scans processed documents and uses an LLM to generate synthetic Q&A pairs (finetune_dataset.jsonl).

Training: A PyTorch script uses LoRA (Low-Rank Adaptation) to fine-tune a Qwen2-0.5B model on this dataset.

Deployment: The trained adapter is merged with the base model at runtime for inference.

3.3 "Smart" Admin Panel

An integrated administrative interface allows non-technical staff to:

Upload documents (.pdf, .docx, .xlsx).

Categorize data dynamically (creating new product lines on the fly).

Perform a "Factory Reset" to wipe all knowledge and retrain from scratch.

4. Engineering Challenges & Solutions

During development, we encountered several critical engineering hurdles. Here is how we solved them:

Challenge 1: Database Concurrency Locks (sqlite3 Read-Only Error)

Problem: The RAG agent crashed when uploading files while the server was running. Both the server process and the ingestion script tried to hold a persistent connection to ChromaDB, locking the sqlite3 file.

Solution: We refactored the entire database layer to be stateless. We removed global database objects. Now, every API request creates a temporary connection, performs its operation, and closes it immediately. This ensures thread safety and prevents locks.

Challenge 2: Hardware Constraints for Fine-Tuning

Problem: Training a Language Model usually requires massive Enterprise GPUs. Our environment was a standard laptop (MacBook Air), causing CUDA errors and OOM (Out of Memory) crashes.

Solution:

Switched from Phi-3 (3.8B params) to Qwen2-0.5B (0.5B params).

Replaced bitsandbytes (NVIDIA-only) with native PyTorch bfloat16 loading.

Utilized LoRA (Low-Rank Adaptation) to train only <1% of parameters, making training feasible on consumer hardware.

Challenge 3: Ambiguous Routing

Problem: The router often confused "product suggestions" (RAG task) with "product searches" (Web task).

Solution: We implemented Chain-of-Thought Prompting in the MCP router. We explicitly defined the scope of each agent in the system prompt (e.g., "RAG is the primary source for specifications; Action is ONLY for blog posts"), drastically improving routing accuracy.

5. Future Scope & Enhancements

To scale Genilia for enterprise production, the following enhancements are proposed:

Vector Database Scaling: Migrate from local ChromaDB to a managed solution like Pinecone or Weaviate to support millions of documents with lower latency.

Redis for Memory: Replace the in-memory Python dictionary in the MCP server with Redis. This allows the application to be stateless and horizontally scalable (running multiple MCP instances behind a load balancer).

Multimodal Capabilities: Integrate Llava or GPT-4o to allow users to upload images of broken products for troubleshooting.

RLHF (Reinforcement Learning from Human Feedback): Add a "thumbs up/down" button to the UI. Use this feedback data to further fine-tune the router and the SLM, creating a self-improving system.

Docker & Kubernetes: Containerize the 4 services into Docker images and orchestrate them via Kubernetes for auto-scaling based on traffic load.

6. Conclusion

Genilia represents a shift from static chatbots to dynamic, agentic AI. By combining the immediacy of RAG, the precision of Fine-Tuning, and the versatility of Web Search, it offers a comprehensive support solution. The successful implementation of this local-first architecture proves that powerful, domain-specific AI can be built cost-effectively without relying on external black-box APIs.