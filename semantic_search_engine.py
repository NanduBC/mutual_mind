import os
import time

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from llamaapi import LlamaAPI
from langchain.schema import SystemMessage
from langchain_experimental.llms import ChatLlamaAPI

from entity_extractor import extract_fund_entities

semantic_search_engine = None

def get_semantic_search_engine():
    global semantic_search_engine
    if semantic_search_engine is None:
        semantic_search_engine = SemanticSearchEngine()
    return semantic_search_engine

class SemanticSearchEngine:
    def __init__(self):
        print('Loading embedding model')
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print('Embedding model loaded')

        print('Loading vector store')
        self.vector_store = Chroma(
            persist_directory="./mutual_funds_store",
            embedding_function=self.embedding_model,
            collection_name="funds-categorical")
        print('Vector store loaded')
    
    def get_similar_documents(self, query):
        # print('Vector Store #docs:', vector_store._collection.count())
        # entities = extract_fund_entities(query)
        # combined_results = []
        # for item in entities:
        #     fund_name = item['fund_name']
        #     results = self.vector_store.similarity_search(fund_name, k=3)
        #     combined_results.extend(results)
        # return combined_results
        return self.vector_store.similarity_search(query, k=3)

    def get_rag_response(self, query):
        # retrieved_docs = vector_store.similarity_search(query, k=5)
        retrieved_docs = self.get_similar_documents(query)
        context = 'Document\n' + '\n\nDocument\n'.join([doc.page_content for doc in retrieved_docs])
        
        system_prompt = """
You are an intelligent semantic search engine that would get the right information about the right funds.
Context: {}
Now the output should be human-readable text which will be used by MF salesperson and should be to the point. Do not include any additional info than being asked for. Provide correct info from context if possible, else respond saying it's not possible.
Input: {}
""".format(context, query)
        llama_client = LlamaAPI(os.environ['LLAMA_API_KEY'])
        llm = ChatLlamaAPI(client=llama_client, model='llama3-8b', temperature=0)
        message = SystemMessage(content=system_prompt)
        response = llm.invoke([message])
        return context, response.content

if __name__ == '__main__':
    semantic_search_engine = get_semantic_search_engine()
    print('Semantic Search Engine. Type "Stop" to stop the search engine')
    while True:
        print('Type in your query:', end=' ')
        query = input()
        if query.lower() == 'stop':
            break
        start_time = time.time()
        context, response = semantic_search_engine.get_rag_response(query)
        end_time = time.time()
        print(response)
        print()
        print('Context:', context)
        print('Time taken:', end_time-start_time)
