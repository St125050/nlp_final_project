import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="AI Caption Generator", page_icon="📝", layout="centered")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load captions
@st.cache_data
def load_captions():
    with open("captions.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_caption_model")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.load_state_dict(torch.load("./fine_tuned_caption_model/model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

# Set up FAISS index
@st.cache_resource
def setup_faiss_index(captions):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    caption_embeddings = embedder.encode(captions)
    dimension = caption_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(caption_embeddings)
    return embedder, index

# Retrieve relevant content
def retrieve_relevant_content(query, embedder, index, captions, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [captions[i] for i in indices[0]]

# Generate caption
def generate_caption(prompt, model, tokenizer, embedder, index, captions):
    trends = ["#Vibes", "#Goals", "#Explore"]
    retrieved = retrieve_relevant_content(prompt, embedder, index, captions)
    cot_prompt = f"Step 1: Start with '{prompt}'.\nStep 2: Add trends {trends}.\nStep 3: Create a catchy caption.\nOutput:"
    inputs = tokenizer(cot_prompt + " " + " ".join(retrieved), return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=30,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95
        )
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption

# Streamlit UI
def main():
    st.title("AI Caption Generator")
    st.markdown("Enter a prompt to generate a catchy caption for your social media post!")

    # Load resources
    with st.spinner("Loading model and resources..."):
        captions = load_captions()
        model, tokenizer = load_model_and_tokenizer()
        embedder, index = setup_faiss_index(captions)

    # Input prompt
    prompt = st.text_input("Enter your prompt (e.g., 'Enjoying a sunny day'):", "")
    
    # Generate button
    if st.button("Generate Caption"):
        if prompt.strip():
            with st.spinner("Generating caption..."):
                try:
                    caption = generate_caption(prompt, model, tokenizer, embedder, index, captions)
                    st.success("Caption generated!")
                    st.write("**Generated Caption:**")
                    st.markdown(f"*{caption}*")
                except Exception as e:
                    st.error(f"Error generating caption: {str(e)}")
        else:
            st.warning("Please enter a prompt.")

    # Sidebar with info
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a fine-tuned DistilGPT-2 model with RAG to generate catchy captions. "
        "Enter a prompt and let the AI create a caption with trendy hashtags!"
    )

if __name__ == "__main__":
    main()