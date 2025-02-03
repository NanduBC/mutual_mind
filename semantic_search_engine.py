import os
import time

import pandas as pd
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
    def __init__(
            self,
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            doc_vector_store_persist_directory='./mutual_funds_store',
            doc_vector_store_collection_name='funds-categorical',
            col_vector_store_persist_directory='./mf_cols',
            col_vector_store_collection_name='cols-store'):
        print('Loading embedding model')
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print('Embedding model loaded')

        print('Loading vector stores')
        self.vector_store = Chroma(
            persist_directory=doc_vector_store_persist_directory,
            embedding_function=self.embedding_model,
            collection_name=doc_vector_store_collection_name)
        self.col_store = Chroma(
            persist_directory=col_vector_store_persist_directory,
            embedding_function=self.embedding_model,
            collection_name=col_vector_store_collection_name)
        print('Vector stores loaded')

        # TODO: Replace dataframe to SQL as storage and retrieval for fund attributes
        self.mutual_fund_data = pd.read_csv('MutualFunds.csv').set_index('fund_symbol')
    
    def get_similar_documents(self, query):
        entities = extract_fund_entities(query)
        combined_results = []
        for item in entities: # Ideally only 1 fund per query
            fund_name = item['fund_name']
            
            mapped_fund_attribute_keys = ['fund_long_name']
            for fund_attribute in item['fund_attributes']:
                mapped_fund_attribute_key = self.col_store.similarity_search(fund_attribute['key'], k=1)[0].page_content
                mapped_fund_attribute_keys.append(mapped_fund_attribute_key)

            retrieved_fund_symbols = [doc.metadata['fund_symbol'] for doc in self.vector_store.similarity_search(fund_name, k=5)]
            # print(retrieved_fund_symbols)
            item_results = []
            for fund_symbol in retrieved_fund_symbols:
                relevant_info = self.mutual_fund_data.loc[fund_symbol][mapped_fund_attribute_keys].to_dict()
                item_results.append(str(relevant_info))
            combined_results.append(item_results)
        # print(combined_results)
        return combined_results

    def get_rag_response(self, query):
        retrieved_docs = self.get_similar_documents(query)
        context = '\n'.join([doc for item in retrieved_docs for doc in item])
        
        system_prompt = """
You are an intelligent semantic search engine that would get the right information about the right funds.
Context: {}
Now the output should be human-readable text which will be used by MF salesperson and should be to the point. Do not include any additional info than being asked for. Provide correct info from context if possible, else respond saying it's not possible.
Input: {}
""".format(context, query)
        llama_client = LlamaAPI(os.environ['LLAMA_API_KEY'])
        llm = ChatLlamaAPI(client=llama_client, model='llama3-70b', temperature=0)
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
        print('Time taken:', end_time-start_time)
