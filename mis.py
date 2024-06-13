# Philippe Joly
# 2024-06-12
# this is a test implementation of mistral 7b using the langchain interface

from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEndpoint as hh
import os


def main():
    load_dotenv(find_dotenv())
    apiKey = os.environ.get('HUGGING_FACE_KEY')
    if not apiKey:
        print("cannot load HUGGING_FACE_KEY")
        return
    print("Keys locked and loaded!")
    try:
        model = hh(repo_id="PygmalionAI/pygmalion-350m",
                   huggingfacehub_api_token=apiKey)        
        response = model.invoke("hey what's up?")
        print("Model Response: ", response)
    except Exception as e:
        print("an error occured: ", str(e))


if __name__ == "__main__":
    main()
