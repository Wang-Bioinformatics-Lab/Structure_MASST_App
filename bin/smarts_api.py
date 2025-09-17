import json
import time

import requests


def query_smarts(smarts, api_key, job_id="default_job", file_format="png"):
    """
    Simple function to query the SMARTS API.

    Args:
        smarts (str): The SMARTS pattern string
        api_key (str): API key for authentication
        job_id (str): Job ID for the request
        file_format (str): Output format (default: "png")

    Returns:
        dict: JSON response from the API
    """
    url = "https://api.smarts.plus/smartsView/"
    data = {
        "job_id": job_id,
        "query": {
            "smarts": smarts,
            "parameters": {
                "file_format": file_format,
                "legend_mode": "3"
            }
        }
    }
    headers = {"Content-Type": "application/json",
               "X-API-Key": api_key}
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(f"Initial SMARTSPlus request failed with status code {response.status_code}. Retrying...")
        time.sleep(2)  # Wait before retrying
        response = requests.post(url, headers=headers, data=json.dumps(data))

    return response.json()
