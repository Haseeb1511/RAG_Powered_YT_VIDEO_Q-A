import streamlit as st
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

load_dotenv()

st.title("🎥 YouTube Video Q&A")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Input YouTube link
url = st.sidebar.text_input("Paste YouTube Link Here:")

@st.cache_data(show_spinner="Fetching transcript...")
def fetch_transcript(video_id):
    try:
        transcript_load = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript_join = " ".join(text["text"] for text in transcript_load)
        return transcript_join
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Creating vector database...")
def create_vectorstore(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.create_documents([transcript_text])
    embedding = load_embedding_model()
    vector_store = FAISS.from_documents(documents=chunks, embedding=embedding)
    return vector_store

if url:
    video_id = url_cleaning(url)
    transcript_text = fetch_transcript(video_id)

    if transcript_text:
        vector_store = create_vectorstore(transcript_text)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

        prompt = PromptTemplate(
            template="Use the given piece of information in {context} to find the answer to the {question}. If you don't know the answer, just say you don't know. Don't try to make it up.",
            input_variables=["context", "question"]
        )

        model = ChatGroq(model="llama-3.1-8b-instant", max_tokens=256)

        chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

        query = st.chat_input("Ask something about the video:")
        if query:
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            result = chain.invoke({"question": query})
            response = result["result"]

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.error("⚠️ Transcript not available for this video.")



# if __name__=="__main__":
#     main()

# GjczwkqsiFk  