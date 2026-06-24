import streamlit as st
from summarizer import TextSummarizer

# Page Configuration
st.set_page_config(page_title="AI Text Summarizer", page_icon="📝")

@st.cache_resource
def load_model():
    return TextSummarizer()

def main():
    st.title("📝 Transformer-Based Abstractive Summarizer")
    st.markdown("AI-powered abstractive text summarization using Transformer models.")
    
    # Initialize Model
    with st.spinner("Loading Transformer Model... Please wait."):
        summarizer = load_model()

    # User Input Section
    st.subheader("Input Text")
    input_text = st.text_area("Paste your long article or paragraph here:", height=300)

    # Sidebar Controls
    st.sidebar.header("Settings")
    max_l = st.sidebar.slider("Max Summary Length", 50, 500, 200)
    min_l = st.sidebar.slider("Min Summary Length", 20, 200, 50)

    if st.button("Summarize"):
        if input_text.strip():
            with st.spinner("Generating Summary..."):
                try:
                    result = summarizer.summarize(input_text, max_len=max_l, min_len=min_l)
                    
                    st.subheader("Generated Summary")
                    st.success(result)
                    
                    # Display Stats
                    st.info(f"Original Length: {len(input_text.split())} words | "
                            f"Summary Length: {len(result.split())} words")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text first!")
    st.markdown("---")
    st.write("Built using Streamlit, Hugging Face Transformers, and BART.")
    
if __name__ == "__main__":
    main()