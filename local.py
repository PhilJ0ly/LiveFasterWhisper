# Philippe Joly
# 2024-06-13

# Test how to run a model locally

from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFaceEndpoint as hf, HuggingFacePipeline
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
hfkey = os.environ.get("HUGGING_FACE_KEY")

# template


template = """question: {question}

Answer: one word answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])


# llm=hf(repo_id="mistralai/Mistral-7B-Instruct-v0.3", huggingfacehub_api_token=hfkey)

# print(llm.invoke(prompt.format(question="What is up?")))
model_id= "facebook/blenderbot-1B-distill"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=100)

local_llm =  HuggingFacePipeline(pipeline=pipe)

print(local_llm.invoke("What is up?"))
