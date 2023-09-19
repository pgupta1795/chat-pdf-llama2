# PDF chatbot using llama2 model

## Pre-requisites

- ### Access llama2 model from meta after approval from huggingface (7b or 13b or 70b) : <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>

- ### Create a read token from here : <https://huggingface.co/settings/tokens>

- ### execute huggingface-cli login and provide read token

- ### If unable to access llama2 model from official meta page then use alternatives such as : <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML>

## Run using following commmands

- ### python ingest.py

- ### python model.py and then ask the question from PDF

  OR

- ### If using streamlit code execute : streamlit run model.py OR If using chainlit code execute : chainlit run model.py
