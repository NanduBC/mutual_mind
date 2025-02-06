import os
import time

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from llamaapi import LlamaAPI
from langchain.schema import SystemMessage
from langchain_experimental.llms import ChatLlamaAPI

from entity_extractor import extract_fund_entities
from logger import get_logger


EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

semantic_search_engine_obj = None
def get_semantic_search_engine():
    '''
    Returns the global object for SemanticSearchEngine.
    Global object pattern is the pythonic variant of Singleton design pattern
    '''
    global semantic_search_engine_obj
    if semantic_search_engine_obj is None:
        semantic_search_engine_obj = SemanticSearchEngine()
    return semantic_search_engine_obj

class SemanticSearchEngine:
    '''
    A semantic search engine for getting right information about right Mutual funds

    Attributes:
    ----------
    vector_store: Vector store for retreiving similar documents to the
     fund extracted
    col_store: Vector store for retrieving similar column name of the
     dataframe given extracted fund attribute
    mutual_funds_data: In-memory store of the dataframe indexed on
     `fund_symbol` for near constant retrieval
    '''
    def __init__(
            self,
            doc_vector_store_persist_directory='./mutual_funds_store',
            doc_vector_store_collection_name='funds-categorical',
            col_vector_store_persist_directory='./mf_cols',
            col_vector_store_collection_name='cols-store'):
        self.logger = get_logger('SemanticSearchEngine')
        self.logger.info('Loading embedding model:%s', EMBEDDING_MODEL_NAME)
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.logger.info('Embedding model loaded')

        self.logger.info('Loading vector stores')
        self.vector_store = Chroma(
            persist_directory=doc_vector_store_persist_directory,
            embedding_function=self.embedding_model,
            collection_name=doc_vector_store_collection_name)
        self.col_store = Chroma(
            persist_directory=col_vector_store_persist_directory,
            embedding_function=self.embedding_model,
            collection_name=col_vector_store_collection_name)
        self.logger.info('Vector stores loaded')
        # TODO: Replace dataframe to SQL as storage and retrieval for fund attributes
        self.mutual_fund_data = pd.read_csv('MutualFunds.csv').set_index('fund_symbol')
    
    def get_relevant_documents(self, doc_query):
        '''
        Returns relevant documents to the `doc_query`. This is done by
        extracting fund entities like fund_name and fund_attributes and
        finding documents that are similar to fund_name and cols
        similar to keys in fund_attribute.

        Parameters:
        ----------
        doc_query: User query about details of some fund

        Returns:
        --------
        A list of documents for each fund name extracted with relevant
        fund attributes present
        '''
        self.logger.info('Relevant document retreival started')
        entities = extract_fund_entities(doc_query)
        self.logger.info('Entities:%s', entities)
        combined_results = []
        for item in entities:
            fund_name = item['fund_name']

            mapped_fund_attribute_keys = ['fund_long_name']
            for fund_attribute in item['fund_attributes']:
                mapped_fund_attribute_key = self.col_store.similarity_search(fund_attribute['key'], k=1)[0].page_content
                mapped_fund_attribute_keys.append(mapped_fund_attribute_key)
            try:
                retrieved_fund_symbols = []
                for doc, relevance_score in self.vector_store.similarity_search_with_relevance_scores(fund_name, k=5):
                    self.logger.info('Relevance score for %s is %s ', doc.page_content, relevance_score)
                    retrieved_fund_symbols.append(doc.metadata['fund_symbol'])
            except AttributeError:
                self.logger.exception('Error faced while fetching relevant documents')
                return combined_results
            item_results = []
            for fund_symbol in retrieved_fund_symbols:
                relevant_info = self.mutual_fund_data.loc[fund_symbol][mapped_fund_attribute_keys].to_dict()
                item_results.append(str(relevant_info))
            combined_results.append(item_results)
        self.logger.info('Relevant document retrieval finished')
        return combined_results

    def get_context_aware_response(self, query:str):
        '''
        Returns context-aware response to the query by using
        approapriate knowledge base

        Parameters:
        ----------
        query: User query regarding details of funds

        Returns:
        -------
        Response from an LLM based on the context provided by the query
        '''
        self.logger.info('Retrieval-augment response generation started')
        retrieved_docs = self.get_relevant_documents(query)
        context = '\n'.join([doc for item in retrieved_docs for doc in item])

        system_prompt = f"""
You are an intelligent and useful semantic search engine that would get the right information about the right funds.
Context: {context}
Now the output should be human-readable text which should be based on the above context only and should be as concise as possible unless specified otherwise.
Do not include any additional info than being asked for.
Provide correct info from context if possible, else respond saying it's not possible without explictly mentioning the context
Input: {query}"""

        llama_client = LlamaAPI(os.environ['LLAMA_API_KEY'])
        llm = ChatLlamaAPI(client=llama_client, model='llama3-70b', temperature=0)
        message = SystemMessage(content=system_prompt)
        response = llm.invoke([message])
        print('Retrieval-augment response generation started')
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
        query_context, query_response = semantic_search_engine.get_context_aware_response(query)
        end_time = time.time()
        print('*****************Query Context******************')
        print(query_context)
        print('************************************************')
        print('Your search result is:', query_response)
        print()
        print('Time taken:', end_time-start_time)
