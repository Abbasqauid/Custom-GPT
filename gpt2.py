import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import LLMChain

## Function to get response from LLaMA 2 model (CTransformers)
def getLLamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(
        model=r"D:\University Projects\KRR 5TH\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={"max_new_tokens": 256, "temperature": 1},
    )

    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'], template=template)

    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

## Function to get response from Hugging Face models (e.g., GPT-2, GPT-Neo, DistilGPT2, MiniLM, etc.)
def get_hf_response(model_name, input_text, task_type):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Check task type and generate corresponding prompt
    if task_type == "Question Answering":
        prompt = f"Answer the following question based on the input: {input_text}"
    elif task_type == "Text Generation":
        prompt = f"Generate a text based on the input: {input_text}"
    else:
        raise ValueError("Invalid task type. Choose either 'Question Answering' or 'Text Generation'.")

    # Encode input and generate response
    inputs = tokenizer(prompt, return_tensors="pt")

    # Adjust generation parameters to reduce repetition and improve coherence
    outputs = model.generate(
        **inputs,
        max_length=150,  # Adjust the length based on your need
        temperature=0.7,  # Adjust for randomness and creativity
        top_p=0.9,  # Nucleus sampling (reduces repetitions)
        do_sample=True,  # Enables sampling
        top_k=50  # Controls the sampling diversity
    )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.set_page_config(page_title="Custom GPT with Multiple Models", layout='centered')
st.header("Custom GPT with Multiple Models")

# User input
input_text = st.text_input("Enter the Input Text")
task_type = st.selectbox("Select Task Type", ["Question Answering", "Text Generation"])

# Adding small models to the model selection dropdown
model_choice = st.selectbox("Select a Model", [
    "GPT-2", "DistilGPT-2", "MiniLM", "T5-Small", "DistilBERT", "BART-Small",
    "ALBERT", "MobileBERT", "ELECTRA-Small", "GPT-Neo", "LLaMA-2"
])

generate = st.button("Generate")

if generate:
    # Select model based on user choice
    if model_choice == "GPT-2":
        response = get_hf_response("gpt2", input_text, task_type)
    elif model_choice == "DistilGPT-2":
        response = get_hf_response("distilgpt2", input_text, task_type)
    elif model_choice == "MiniLM":
        response = get_hf_response("Microsoft/MiniLM-L12-H384-uncased", input_text, task_type)
    elif model_choice == "T5-Small":
        response = get_hf_response("t5-small", input_text, task_type)
    elif model_choice == "DistilBERT":
        response = get_hf_response("distilbert-base-uncased", input_text, task_type)
    elif model_choice == "BART-Small":
        response = get_hf_response("facebook/bart-small", input_text, task_type)
    elif model_choice == "ALBERT":
        response = get_hf_response("albert-base-v2", input_text, task_type)
    elif model_choice == "MobileBERT":
        response = get_hf_response("google/mobilebert-uncased", input_text, task_type)
    elif model_choice == "ELECTRA-Small":
        response = get_hf_response("google/electra-small-discriminator", input_text, task_type)
    elif model_choice == "GPT-Neo":
        response = get_hf_response("EleutherAI/gpt-neo-1.3B", input_text, task_type)
    elif model_choice == "LLaMA-2":
        response = getLLamaresponse(input_text, 100, 'General')  # Adjust parameters as needed

    # Display the response
    st.write(response)
