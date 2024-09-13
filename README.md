# AI Resume Screening Agent

Welcome to the AI Resume Screening Agent repository! This project provides a powerful AI-driven tool designed to streamline the process of screening and evaluating resumes. Using state-of-the-art machine learning models, the agent can assess candidates based on their resumes across various categories and provide detailed evaluations.

## Features

- **Resume Screening**: Automatically analyze and evaluate resumes based on predefined categories.
- **Customizable Evaluation Criteria**: Configure evaluation criteria and weightings to match your specific needs.
- **API Integration**: Easily integrate the AI agent with other systems via API endpoints.
- **Environment Variable Management**: Securely manage sensitive information such as API keys using environment variables.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- [Python 3.8+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)
- [Python-dotenv](https://pypi.org/project/python-dotenv/) (for environment variable management)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/ai-resume-screening-agent.git
    cd ai-resume-screening-agent
    ```

2. **Install dependencies using Poetry:**

    ```bash
    poetry install
    ```

3. **Set up environment variables:**

    Create a `.env` file in the root directory and add the following environment variables:

    ```plaintext
    QDRANT_URL=https://your-qdrant-url
    QDRANT_API_KEY=your-api-key
    OPENAI_API_KEY=your-openai-api-key
    ```

4. **Run the application:**

    You can run the application using Poetry:

    ```bash
    poetry run python main.py
    ```

## Usage

### Example

Here's a basic example of how to use the AI Resume Screening Agent:

```python
from your_module import ResumeScreeningAgent

# Initialize the agent
agent = ResumeScreeningAgent()

# Provide a resume to screen
resume_path = 'path/to/resume.pdf'
evaluation_result = agent.screen_resume(resume_path)

# Print the results
print(evaluation_result)