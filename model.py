from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


path_of_VS = "vector_store/faiss"
embedding =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.load_local(path_of_VS,embeddings=embedding,allow_dangerous_deserialization=True) 
query = input("User: ")
retriver = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})

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
    retriever=retriver,
    chain_type_kwargs = {"prompt":template})

result = chain.invoke({"query":query})
print(result["result"])

