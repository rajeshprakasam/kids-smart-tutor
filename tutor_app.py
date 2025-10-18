import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the GPT-2 model and tokenizer (much faster and smaller than Phi-3)
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Fast and lightweight (~124M parameters)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

# Generate response using GPT-2 model, tailored to grade level
def generate_response(question, grade, tokenizer, model):
    # Simple prompt for GPT-2 (no special chat template)
    prompt = f"Question: {question}\n\nAnswer in simple, easy-to-understand words for a {grade}th grade student. Keep it fun and clear.\n\nAnswer:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=200,  # Shorter for speed
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract the generated response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Get only the new part after the prompt
    assistant_response = full_response[len(prompt):].strip()
    return assistant_response if assistant_response else "Hmm, let me think... That's a great question! The answer is like a fun adventure in learning."

# Main Streamlit app
def main():
    st.set_page_config(page_title="Kids' Smart Tutor", page_icon="🎓", layout="wide")
    st.title("🎓 Kids' Smart Tutor")
    st.markdown("Ask me anything about school subjects! I'll explain it just right for your grade.")

    # Load model once
    try:
        tokenizer, model = load_model()
        st.success("Tutor is ready! Model loaded successfully (GPT-2 for extra speed!).")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure you have transformers and torch installed: `pip install streamlit transformers torch`")
        st.stop()

    # Grade selection
    grade = st.selectbox("What grade are you in?", options=list(range(1, 13)), index=5)

    # Question input
    question = st.text_area(
        "What's your question?",
        placeholder="e.g., Why do planets orbit the sun?",
        height=100,
        help="Type your question here and hit Enter or click Ask!"
    )

    # Submit button
    if st.button("Ask the Tutor! 🚀", type="primary"):
        if question.strip():
            with st.spinner("Thinking quickly... Almost there!"):
                response = generate_response(question, grade, tokenizer, model)
                st.subheader("Here's what I found:")
                st.markdown(f"**For a {grade}th grader:** {response}")
                st.balloons()  # Fun animation!
        else:
            st.warning("Please enter a question first!")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by GPT-2 from Hugging Face. Made with ❤️ for fast learning!*")

if __name__ == "__main__":
    main()
