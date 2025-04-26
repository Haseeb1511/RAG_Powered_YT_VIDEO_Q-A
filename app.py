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
st.header("Welcome to the YouTube Video Q&A App! ")
st.markdown("This app allows you to interact with any YouTube video by asking questions and receiving answers based on the video’s content. Powered by Advanced AI and Natural Language Processing, the app extracts the transcript of the video and lets you query it in real time.")
st.markdown("""How it works:
1. Paste any YouTube video link into the sidebar.
2. The app will fetch the transcript (if available) and process it.
3. Ask questions about the video, and the AI will respond with accurate answers using the transcript data.""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])


st.sidebar.write("Ask Anything from YouTube Videos!🚀")
st.sidebar.markdown("**Step 1:** Paste YouTube Link 📺")
st.sidebar.markdown("**Step 1:** Wait a few second and let the app do its magic❤️")
url = st.sidebar.text_input("Paste YouTube Link Here:")
st.sidebar.markdown('<p style="color:red; font-weight: bold; font-size: 24px;">Key Features:</p>', unsafe_allow_html=True)
st.sidebar.markdown('''
1. **Instant Video Transcript Retrieval:** Quickly fetch the transcript of any video with subtitles.
2. **Contextual Q&A:** Ask questions specific to the video and get answers based only on the content within the transcript.
3. **Interactive Experience:** Seamlessly chat with the app to dive deeper into the video’s content.''')


@st.cache_data(show_spinner="Fetching transcript...")
def transcribe_loading(url:str):
    if "watch?v=" not in url:
        return None
    video_id = url.split("=")[1].split("&")[0]
    try:
        transcript_load = YouTubeTranscriptApi.get_transcript(video_id,languages=["en"])
        transcript_join = " ".join(text["text"] for text in transcript_load)
        return transcript_join
    except (TranscriptsDisabled,NoTranscriptFound):
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


vector_store=None
if url:
    transcript = transcribe_loading(url)
    if transcript:
        vector_store = create_vectorstore(transcript)
    else:
        st.error("Transcript is not Avaliable for this !!!!")

if vector_store:
    query = st.chat_input("Ask something about the video:")
    if query:
        retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})


        template = PromptTemplate(
                template="""You are a highly accurate assistant.
            Use ONLY the given context to answer the user's question.
            If the context does not contain the information needed, simply reply:
            "I don't know based on the given context."
            CONTEXT:
            {context}
            QUESTION:
            {question}
            Your Answer:""",
            input_variables=["context", "question"])

        model = ChatGroq(model="llama-3.1-8b-instant",max_tokens=256)

        from langchain.chains import RetrievalQA
        chain = RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            chain_type_kwargs = {"prompt":template})


        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Generating response..."):
            result = chain.invoke({"query": query})
            response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # else:
    #     st.error("⚠️ Transcript not available for this video.")

st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            font-size: 12px;
            color: gray;
        }
    </style>
    <div class="footer">
        Made with ❤️ by Haseeb Manzoor
    </div>
""", unsafe_allow_html=True)

