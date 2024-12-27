import streamlit as st
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    base_model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
    model = PeftModel.from_pretrained(base_model, "abbasquaid/llama1B-fune-tine-model")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

    # Set pad_token_id for compatibility
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()

# Streamlit app UI
st.title(" Welcome To EDU-BOT. ")
st.write("Ask a question, and get a concise, accurate answer.")

# Input text
input_text = st.text_input("Enter your question:", "")

# Generate response
if st.button("Get Answer"):
    if input_text.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        # Tokenize the input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=30,
            truncation=True,
            padding=True
        )

        # Generate the output
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            num_beams=3,
            do_sample=False
        )

        # Decode and process the output
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the input question and extra details
        if decoded_output.lower().startswith(input_text.lower()):
            decoded_output = decoded_output[len(input_text):].strip()

        # Keep only the first sentence
        sentences = decoded_output.split(".")
        processed_output = sentences[0].strip()

        # Display the output
        st.success(f"Answer: {processed_output}")
