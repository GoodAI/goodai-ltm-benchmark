import json
import requests
from time import sleep
from flask import Flask, request

app = Flask("MemGPTproxy")


@app.route("/<path:url_path>", methods=["POST"])
def proxy_request(url_path):
    """
    Be extremely careful when changing anything in this code.
    Following actions leak stuff into the HTTP response:
    1. print
    2. json.loads(response.json())
    """
    openai_api_url = f"https://api.openai.com/{url_path}"
    headers = {k: v for k, v in request.headers.items()}
    headers["Host"] = "api.openai.com:5000"
    data = request.get_data()
    num_tries = 5
    for _ in range(num_tries):
        try:
            response = requests.post(openai_api_url, data=data, headers=headers)
            if response.status_code == 200:
                break
        except:
            pass
        sleep(3)
    headers = {k: v for k, v in response.headers.items()}
    if response.status_code != 200:
        del headers["Content-Length"]
    if response.status_code == 200:
        del headers["Transfer-Encoding"]
        del headers["Content-Encoding"]
        with open("model_interfaces/memgpt-logs.jsonl", "a") as fd:
            fd.write(f"{json.dumps(json.loads(response.text))}\n")
    return response.json(), response.status_code, headers.items()


app.run(port=5000, use_reloader=False)
