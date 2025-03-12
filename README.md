# Multi-Agent RAG System

The **Multi-Agent RAG (Retrieval-Augmented Generation) System** is a powerful framework that leverages the combination of LangChain, Qdrant, Celery, Redis, and asyncio to enable the creation of sophisticated multi-agent systems for efficient retrieval-augmented generation tasks. This project enables distributed querying, retrieval, and response generation, making it highly suitable for complex NLP and AI tasks such as chatbots, question answering systems, and document processing.

## Features

- **Multi-Agent Architecture**: Supports distributed agents working together to perform complex tasks.
- **LangChain Integration**: Simplify the creation of NLP pipelines and workflows with LangChain.
- **Qdrant Vector Search**: Use Qdrant as a vector search engine to enhance retrieval-based tasks.
- **Asynchronous Processing**: Handle high volumes of requests concurrently using asyncio.
- **Task Queue Management**: Manage tasks efficiently with Celery and Redis, ensuring scalable, distributed task execution.

## Installation

To get started with the Multi-Agent RAG system, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/pr0fi7/multi_agent_rag.git
   cd multi_agent_rag

2. **Start docker container**:
  ```bash
  docker-compose up --build
  ```

## Usage

The Multi-Agent RAG system provides endpoints for interacting with multiple agents, managing retrieval tasks, and performing generation tasks. You can customize the agents' behavior and retrieval strategies based on your project requirements.

## Contributing

Contributions to the Multi-Agent RAG system are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes.
4. Commit your changes with clear, descriptive messages.
5. Push your changes to your forked repository.
6. Submit a pull request detailing your changes.

Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

The Multi-Agent RAG system is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [LangChain](https://langchain.com/) for facilitating advanced NLP workflows.
- [Qdrant](https://qdrant.tech/) for providing efficient vector search capabilities.
- [Celery](https://docs.celeryproject.org/en/stable/) for asynchronous task queue management.
- [Redis](https://redis.io/) for fast, in-memory data store solutions.

For more information, visit the [GitHub repository](https://github.com/pr0fi7/multi_agent_rag).
