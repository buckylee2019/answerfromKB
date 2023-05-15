import os
import json
from flask import Flask, request, jsonify
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from requests import post
import openai
from json_utils.json_fix_general import correct_json,add_quotes_to_property_names
from utils.chat_utils import ask

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


def query_watson_discovery(query):
    return discovery.query(
        project_id=project_id,
        collection_ids=collection_list,
        natural_language_query=query,
        passages={
            "enabled":True,
            "find_answers":True,
            "per_document":True,
            "count":20,
            "fields":["text"],
            "characters":280,
            "max_per_document":1},
        count=20,
    ).get_result()


def bam_search(query):
    discovery_response = query_watson_discovery(query)


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

def generate_openai_response(messages):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0301",
        messages = messages,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()

def fix_json(json_string):
    try:
        # attempt to parse the string as JSON
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError as e:
        # if there's an error, try to fix it by removing characters before the first open brace or bracket
        pos = e.pos
        while pos > 0:
            pos -= 1
            if json_string[pos] in {'{', '['}:
                fixed_json_string = json_string[pos:]
                try:
                    parsed_json = json.loads(fixed_json_string)
                    return parsed_json
                except json.JSONDecodeError:
                    pass
        # if we can't fix the string, raise an exception
        raise ValueError("Cannot fix JSON string")

@app.route('/openai_watson_discovery_search', methods=['POST'])
def openai_watson_discovery_response():
    # ... (The same code as in the `bam_response` function until the LLM input is prepared)
    data = request.get_json()

    if 'query' not in data:
        return jsonify({"error": "Missing 'query' in request data."}), 400

    query = data['query']

    # Call Watson Discovery

    discovery_response = query_watson_discovery(query)


    inputText = "Question: " + query + "?"
    prompt = "Given the context provided below, answer the following question：" + query + "\n\n Context: \n"
    for passage in discovery_response["results"][:5]:
        prompt += "\n" + passage["document_passages"][0]["passage_text"]
    prompt += inputText
    messages = [
            {"role": "user", "content": prompt}
        ]
    answer = generate_openai_response(messages)
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

@app.route('/openai_search', methods=['POST'])
def openai_response():
    # ... (The same code as in the `bam_response` function until the LLM input is prepared)
    data = request.get_json()

    if 'query' not in data:
        return jsonify({"error": "Missing 'query' in request data."}), 400

    query = data['query']

    inputText = "Question: " + query + "?"
    prompt = "Given the context provided below, answer the following question：" + query + "\n\n Context: \n"

    prompt += inputText
    messages = [
            {"role": "user", "content": prompt}
        ]
    answer = generate_openai_response(messages)
    response_data = {
        "matching_results": 1,
        "retrievalDetails": {
            "document_retrieval_strategy": "untrained"
        },
        "results": [
            {
                "document_id": "NULL",
                "title":"NULL",
                "text": "NULL",
                "link": "NULL",
                "document_passages": [
                    {
                        "passage_text": "NULL",
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

@app.route('/watson_discovery_search', methods=['POST'])
def watson_discovery_response():
    # ... (The same code as in the `bam_response` function until the LLM input is prepared)
    data = request.get_json()

    if 'query' not in data:
        return jsonify({"error": "Missing 'query' in request data."}), 400

    query = data['query']

    # Call Watson Discovery
    discovery_response = query_watson_discovery(query)
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
                            "answer_text": discovery_result["text"]
                            }
                        ]
                    }   
                ]
            }
        ]
    }
    return jsonify(response_data)

@app.route('/orchestrator_search', methods=['POST'])
def orchestrator_response():
    data = request.get_json()
    method = data['method']
    method_dict = {'bam':bam_response,
                'openai_watson_discovery_search':openai_watson_discovery_response,
                'openai_search':openai_response,
                'watson_discovery_search':watson_discovery_response}
    response = method_dict[method]()
    return response

@app.route('/chat', methods=['POST'])
def chat():
    req_data = request.get_json()
    data = req_data['data']
    memory = req_data['memory']

    format = {
        "output": {
            "generic":{
                "text":"Here is plain text response",
                "option":{
                    "title":"title of options",
                    "options": [
                        {
                            "label": "option 1 name",
                            "value": {
                                "input": {
                                    "text": "the text which will sent back as input text to dialog"
                                }
                            }
                        },
                        {
                            "label": "option 2 name",
                            "value": {
                                "input": {
                                    "text": "the text which will sent back as input text to dialog"
                                }
                            }
                        },
                        {
                            "label": "option 3 name",
                            "value": {
                                "input": {
                                    "text": "the text which will sent back as input text to dialog"
                                }
                            }
                        }
                    ]                   
                }             
            }
        }
    }
    prompt = f'You are AGI-Demo, An ai designed to become a tour guide.\n\
    Your decisions must always be made independently without seeking user assistance. \
    Play to your strengths as an LLM and pursue simple strategies with no legal complications.\n\n\
    Dialog type:\n\
    1. Option: Options for user to choose which action to do next, can be more than three options. \
    Note that the text in the options will be sent back to dialog as a input, so this need more details.\n\
    2. Text: Detail explanation of the answer.\n\n\
    Constraints:\n\
    1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.\n\
    2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.\n\
    3. No user assistance.\n4. Make sure option field is not empty and there is always an option \"End conversation\". \n\n\
    Performance Evaluation:\n\
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n\
    2. Constructively self-criticize your big-picture behavior constantly.\n\
    3. Reflect on past decisions and strategies to refine your approach.\n\
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.\n\n\
    You should only respond in JSON format as described below \nResponse Format:" + {str(format)} + "\n\
    Ensure the response can be parsed by Python json.loads \n\
    The current time and date is Fri Apr 28 01:48:03 2023 \n\
    This reminds you of these events from your past: " + {str(memory)}'

    
    messages = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "user",
                "content": "Answer the question: " + data + "\n \
                Determine which dialog to use, and respond using the format specified above (don't need to reply anything else other than JSON):" 
            }
        ]
    
    response = generate_openai_response(messages)
    corrected_json = correct_json(response)

    return corrected_json

@app.route('/vector_search', methods=['POST'])
def vectorsearch_response():
    data = request.get_json()
    query = data['query']
    answer, reference = ask(query)
    response = {
        "answer": answer,
        "reference": reference
    }
    return response


if __name__ == '__main__':
    app.run(port="3000",host='0.0.0.0')