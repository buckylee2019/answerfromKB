import os
import json
from flask import Flask, request, jsonify
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from requests import post
import openai

app = Flask(__name__)

# Set up Watson Discovery credentials
api_key = os.environ.get("WATSON_API_KEY")
url = os.environ.get("WATSON_URL")
project_id = os.environ.get("WATSON_PROJECT_ID")
openai.api_key = os.environ.get("OPENAI_API_KEY")
collection_list = eval(os.environ.get("WD_COLLECTION_IDS"))

authenticator = IAMAuthenticator(api_key)
discovery = DiscoveryV2(version="2021-09-01", authenticator=authenticator)
discovery.set_service_url(url)

# Set up BAM API credentials
bam_api_key = os.environ.get("BAM_API_KEY")
bam_url = "https://bam-api.res.ibm.com/v1/generate"

@app.route('/bam', methods=['POST'])
def process_request():
    data = request.get_json()

    if 'query' not in data:
        return jsonify({"error": "Missing 'query' in request data."}), 400

    query = data['query']

    # Call Watson Discovery
    discovery_response = discovery.query(
        project_id=project_id,
        collection_ids=[""],
        natural_language_query=query,
        count=5
    ).get_result()

    # Generate the input text for BAM API
    input_text = f"Question: {query}?"
    payload = f"Given the context provided below, answer the following question：{query}\n\n Context: \n"
    cnt = 0

    for result in discovery_response['results']:
        if cnt >= 1:
            break

        payload += "\n" + result['text'][0]
        cnt += 1

    payload += input_text

    # Call BAM API
    headers = {"Authorization": f"Bearer {bam_api_key}"}
    bam_request_data = {
        "model_id": "bigscience/bloomz",
        "inputs": [payload],
        "parameters": {
            "decoding_method": "sample",
            "temperature": 0.3,
            "top_p": 1,
            "top_k": 50,
            "random_seed": None,
            "repetition_penalty": None,
            "stop_sequences": None,
            "min_new_tokens": 10,
            "max_new_tokens": 60
        }
    }

    bam_response = post(bam_url, json=bam_request_data, headers=headers)
    bam_response_data = bam_response.json()

    # Process and return the response
    answer = bam_response_data["results"][0]["generated_text"].strip()
    discovery_result = discovery_response["results"][0]

    response_data = {
        "matching_results": 1,
        "retrievalDetails": {
            "document_retrieval_strategy": "untrained"
        },
        "results": [
            {
                "document_id": discovery_result["document_id"],
                "title": discovery_result["title"],
                "text": discovery_result["text"],
                "link": "NULL",
                "document_passages": [
                    {
                        "passage_text": discovery_result["document_passages"][0]["passage_text"],
                        "passageAnswers": [
                        {
                            "answer_text": answer
                            }
                        ]
                    }   
                ]
            }
        ]
    }
    return jsonify(response_data)
def generate_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
@app.route('/openai_search', methods=['POST'])
def openai_response():
    # ... (The same code as in the `bam_response` function until the LLM input is prepared)
    data = request.get_json()

    if 'query' not in data:
        return jsonify({"error": "Missing 'query' in request data."}), 400

    query = data['query']

    # Call Watson Discovery
    discovery_response = discovery.query(
        project_id = project_id,
        collection_ids = collection_list,
        natural_language_query = query,
        count = 5
    ).get_result()

    inputText = "Question: " + query + "?"
    prompt = "Given the context provided below, answer the following question：" + query + "\n\n Context: \n"
    for passage in discovery_response["results"][:3]:
        prompt += "\n" + passage["text"][0]

    prompt += inputText
    answer = generate_openai_response(prompt)
    discovery_result = discovery_response["results"][0]
    response_data = {
        "matching_results": 1,
        "retrievalDetails": {
            "document_retrieval_strategy": "untrained"
        },
        "results": [
            {
                "document_id": discovery_result["document_id"],
                "title": discovery_result["title"],
                "text": discovery_result["text"],
                "link": "NULL",
                "document_passages": [
                    {
                        "passage_text": discovery_result["document_passages"][0]["passage_text"],
                        "passageAnswers": [
                        {
                            "answer_text": answer
                            }
                        ]
                    }   
                ]
            }
        ]
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(port="3000",host='0.0.0.0')