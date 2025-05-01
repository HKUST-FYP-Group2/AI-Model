import os
import requests
import copy
import json_repair


def decimal_to_pentanary(decimal_number):

    pentanary_number = ""
    for _ in range(4):
        remainder = decimal_number % 5
        pentanary_number = str(remainder) + pentanary_number
        decimal_number //= 5

    return pentanary_number


class Qwen_Communicator:
    def __init__(self):
        self.url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
        }

    def get_qwen_data_dict(self, base64Image):
        data = {
            "model": "qwen2.5-vl-32b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": """
                    Analyse the image and assign 2 simple and specific English keywords along with the description of the image for the background music retrieval from Freesound.
                    The keywords should be related to what a person might expect to if they actually were in the location of the image.
                    Some of the areas which you can focus on are:
                        1. Main objects producing sound (e.g., "rain", "fire")
                        2. Environmental context (e.g., "forest", "city")
                        3. Sound characteristics (e.g., "calm", "rhythmic")
                        4. Meteorological elements (e.g., "storm", "windy")
                    Make sure that the keywords returned can be used to get comforting background music but also realistic sounds based on the context of the image.
                    
                    The description should be a brief summary of what can be seen in the image and the tyep of mood it may give.
                    
                    The output should be in JSON format:
                    {
                        "keywords": ["keyword1", "keyword2"],
                        "description": "A brief description of the image."
                    }
    
                    
                    The output should be in the format given above and nothing else should be included
                    """,
                    "attachments": [
                        {
                            "type": "image",
                            "content": f"data:image/jpeg;base64,{base64Image}",
                        }
                    ],
                }
            ],
            "max_tokens": 300,
        }

        return data

    def get_qwen_response(self, base64Image):
        payload = self.get_qwen_data_dict(base64Image)
        local_headers = copy.deepcopy(self.headers)
        local_headers["Authorization"] = os.getenv("QWEN_API_KEY")
        response = requests.post(self.url, headers=local_headers, json=payload)
        if response.status_code != 200:
            return {"keywords": ["", ""], "description": ""}
        response = response.json()
        response = response["choices"][0]["message"]["content"]
        return json_repair.loads(response)
