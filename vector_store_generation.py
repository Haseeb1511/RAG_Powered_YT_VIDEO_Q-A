from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled,NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings




def trancribe_loading(url:str):
    video_id = url.split("=")[1].split("&")[0]
    try:
        transcript_load = YouTubeTranscriptApi.get_transcript(video_id,languages=["en"])
        transcript_join = " ".join(text["text"] for text in transcript_load)
    except (TranscriptsDisabled,NoTranscriptFound):
        return None
    
transcipt = trancribe_loading(url=input("Enter Youtube Video Link:"))

#----------------------------------------Text-Splitting----------------------------------------------------------

def text_splitting():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150)
    chunks= splitter.create_documents([transcipt])
    return chunks
chunks = text_splitting()


#---------------------------------------------VectorStore-----------------------------------------------------------

embedding_model ="sentence-transformers/all-MiniLM-L6-v2"
def embedding_model(model):
    return HuggingFaceEmbeddings(model_name=embedding_model)


vector_store = FAISS.from_documents(documents=chunks,embedding=embedding_model)
vector_store.save_local("vector_store/faiss")

