from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="./mutual_funds_store",
    embedding_function=embedding_model,
    collection_name="funds-categorical")
print('Vector Store #docs:', vector_store._collection.count())

print('Enter query:', end=' ')
query = input()

# Perform a similarity search
results = vector_store.similarity_search("Franklin", k=5)
for doc in results:
    print(doc.page_content, doc.metadata)
