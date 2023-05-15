from typing import Any, Dict
import requests
import os
import pandas as pd
import re
import os
import json
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from requests import post
import openai



SEARCH_TOP_K = 20
# Set up Watson Discovery credentials
api_key = os.environ.get("WATSON_API_KEY")
url = os.environ.get("WATSON_URL")
project_id = os.environ.get("WATSON_PROJECT_ID")
collection_list = eval(os.environ.get("WD_COLLECTION_IDS"))
DATABASE_BEARER_TOKEN = os.environ.get("DATABASE_BEARER_TOKEN")
authenticator = IAMAuthenticator(api_key)
discovery = DiscoveryV2(version="2021-09-01", authenticator=authenticator)
discovery.set_service_url(url)

# Set up BAM API credentials


def query_watson_discovery(query):
    return discovery.query(
        project_id=project_id,
        collection_ids=collection_list,
        natural_language_query=query,
        passages={
            "enabled":True,
            "find_answers":True,
            "per_document":True,
            "count":50,
            "fields":["text"],
            "characters":280,
            "max_per_document":1},
        count=50
        
    ).get_result()
    
def upsert_file(directory: str):
    """
    Upload all files under a directory to the vector database.
    """
    url = "http://162.133.130.0:8000/upsert-file"
    headers = {"Authorization": "Bearer " + DATABASE_BEARER_TOKEN}
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as f:
                file_content = f.read()
                files.append(("file", (filename, file_content, "text/plain")))
            response = requests.post(url,
                                     headers=headers,
                                     files=files,
                                     timeout=600)
            if response.status_code == 200:
                print(filename + " uploaded successfully.")
            else:
                print(
                    f"Error: {response.status_code} {response.content} for uploading "
                    + filename)


def upsert(row: dict):
    """
    Upload one piece of text to the database.
    """

    if row["pinecone"]:
        return
    url = "http://162.133.130.0:8000/upsert"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + DATABASE_BEARER_TOKEN,
    }

    data = {
        "documents": [{
            "id": row["id"],
            "text": row["content"],
        }]
    }
    response = requests.post(url, json=data, headers=headers, timeout=600)

    if response.status_code == 200:
        print("uploaded successfully.")
    else:
        print(f"Error: {response.status_code} {response.content}")


def query_database(query_prompt: str) -> Dict[str, Any]:
    """
    Query vector database to retrieve chunk with user's input question.
    """
    url = "http://162.133.130.0:8000/query"
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json",
        "Authorization": f"Bearer {DATABASE_BEARER_TOKEN}",
    }
    data = {"queries": [{"query": query_prompt, "top_k": SEARCH_TOP_K}]}

    response = requests.post(url, json=data, headers=headers, timeout=600)

    if response.status_code == 200:
        result = response.json()
        # process the result
        return result
    else:
        raise ValueError(f"Error: {response.status_code} : {response.content}")


if __name__ == "__main__":
    wd_result = {"id":[],"title":[],"content":[],'pinecone':[]}
    result = query_watson_discovery("陳玉珠犯罪")
    
    for r in result["results"]:
        wd_result["id"].append(r["document_id"])
        wd_result["content"].append(re.sub(r"<em>|</em>","",r["document_passages"][0]["passage_text"]))
        wd_result["pinecone"].append(False)
        wd_result['title'].append(r["title"])
    wd_pd = pd.DataFrame(wd_result).drop_duplicates(subset=['content','title'])
    wd_pd = wd_pd.loc[:, ['id', 'content', 'pinecone']]
    
    query_result = query_database("陳玉珠犯罪")
    for r in query_result["results"][0]["results"]:
        new_row = {'id': re.sub(r"_[0-9]$", "", r["id"]), 'content': r["text"],'pinecone':True}
        wd_pd = wd_pd.append(new_row, ignore_index=True)
    wd_pd.drop_duplicates(subset=['id'], inplace=True, keep=False)
    wd_pd.drop_duplicates(subset=['content'], inplace=True)
    wd_pd.apply(upsert,axis = 1)
    