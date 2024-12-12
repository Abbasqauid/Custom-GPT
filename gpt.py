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

## Function to get response from Hugging Face models (e.g., GPT-2, GPT-Neo, Falcon)
def get_hf_response(model_name, input_text, no_words, blog_style):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Format the prompt
    prompt = f"Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words."

    # Encode input and generate response
    inputs = tokenizer(prompt, return_tensors="pt")

    # Adjust generation parameters to reduce repetition and improve coherence
    outputs = model.generate(
        **inputs,
        max_length=256,  # Adjust the length based on your need
        temperature=0.9,  # More creative, but adjust to balance randomness
        top_p=0.9,  # Nucleus sampling (reduces repetitions)
        do_sample=True,  # Enables sampling, instead of greedy decoding
        top_k=50  # Controls the sampling diversity (higher means more options)
    )

    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.set_page_config(page_title="Custom GPT With Multiple Models", layout='centered')
st.header("Custom GPT With Multiple Models")

# User input
input_text = st.text_input("Enter the text here")
no_words = st.text_input("No of Words")
blog_style = st.selectbox("Writing the blog for", ("Researchers", "Data Scientist", "Common People"))
model_choice = st.selectbox("Select a Model", ["LLaMA-2", "Falcon-7B", "GPT-2", "GPT-Neo"])

generate = st.button("Generate")

if generate:
    # Select model based on user choice
    if model_choice == "LLaMA-2":
        response = getLLamaresponse(input_text, no_words, blog_style)
    elif model_choice == "Falcon-7B":
        response = get_hf_response("tiiuae/falcon-7b", input_text, no_words, blog_style)
    elif model_choice == "GPT-2":
        response = get_hf_response("gpt2", input_text, no_words, blog_style)
    elif model_choice == "GPT-Neo":
        response = get_hf_response("EleutherAI/gpt-neo-1.3B", input_text, no_words, blog_style)

    # Display the response
    st.write(response)
