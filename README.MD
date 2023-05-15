# AI Question Answering API

This repository contains a Flask-based RESTful API that generates answers to user queries using IBM Watson Discovery as the knowledge base and two different models for generating the answers: IBM Research foundation model (BAM) and OpenAI.

The API has two endpoints, one for each model: `/bam` for the IBM Research foundation model (BAM) and `/openai_search` for the OpenAI model.

## Getting Started

These instructions will help you set up and run the AI Question Answering API on your local machine.

### Prerequisites

You will need the following software installed on your machine:

- Python 3.6 or later
- pip (Python package manager)
- virtualenv (optional, but recommended)

### Installation

1. Clone the repository:

    ```git clone https://github.com/buckylee2019/ai-question-answering-api.git```


1. (Optional, but recommended) Create a virtual environment and activate it:

    ```
    cd ai-question-answering-api
    python3 -m venv venv
    source venv/bin/activate # On Windows, use venv\Scripts\activate
    ```

3. Install the required Python packages:

    ```pip install -r requirements```

4. Set up vector database interface by cloning the chatgpt 

    ### Quickstart

    Follow these steps to quickly set up and run the ChatGPT Retrieval Plugin:

    1. Install Python 3.10, if not already installed.
    2. Clone the repository: `git clone https://github.com/openai/chatgpt-retrieval-plugin.git`
    3. Navigate to the cloned repository directory: `cd /path/to/chatgpt-retrieval-plugin`
    4. Install poetry: `pip install poetry`
    5. Create a new virtual environment with Python 3.10: `poetry env use python3.10`
    6. Activate the virtual environment: `poetry shell`
    7. Install app dependencies: `poetry install`
    8. Create a [bearer token](#general-environment-variables)
    9. Run the API locally: `poetry run start`

5. Set up the necessary API keys and credentials as environment variables:

    ```
   sh variables.sh
    ```

6. Make sure update vector database, using your own documents.

## Run the Flask application:
```python app.py```


The API should now be running on `http://localhost:3000`.

## API Usage

You can use tools like `curl` or Postman to send requests to the API. Below are examples of how to send requests to each endpoint:

### IBM Research foundation model (BAM)

Send a POST request to `/bam` with a JSON payload containing the query:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "What is AI?"}' http://localhost:3000/bam
```
### OpenAI

Send a POST request to /openai with a JSON payload containing the query:

```
bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "What is AI?"}' http://localhost:3000/openai_search 
```

### Search using watson disovery as knowledge base

```
bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "誰是吳宗憲"}' http://localhost:3000/openai_watson_discovery_search

```
### Search using watson disovery as knowledge base and reranking using vector database
```
bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "誰是吳宗憲"}' http://localhost:3000/vector_search
```
### Advanced Usage

Connect to Watson Assistant, you can use the OpenAPI.json to add your own extension.

For Model in BAM 
![bam](https://github.com/buckylee2019/answerfromKB/blob/main/screenshot/bloom.png)

For Model in OpenAI
![openai](https://github.com/buckylee2019/answerfromKB/blob/main/screenshot/chatgpt.png)


### Tips 

In case, you don't have any server to use, you can always count on ngrok which expose your port for external use.

## On MacOS
```
brew install ngrok -cask 
```

Expose your port:

``` 
    ngrok http 8080
```
### License

This project is licensed under the MIT License