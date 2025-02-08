import os
import time
import yaml

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from llamaapi import LlamaAPI
from langchain.schema import SystemMessage
# from langchain_experimental.llms import ChatLlamaAPI
from langchain_openai import ChatOpenAI

from entity_extractor import extract_fund_entities
from logger import get_logger

with open('embedding_model_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

EMBEDDING_MODEL_NAME = config['embedding_model']['name']
MIN_RELEVANCE_THRESHOLD = 0.5

semantic_search_engine_obj = None
def get_semantic_search_engine():
    '''
    Returns the global object for SemanticSearchEngine.
    '''
    global semantic_search_engine_obj
    if semantic_search_engine_obj is None:
        semantic_search_engine_obj = SemanticSearchEngine()
    return semantic_search_engine_obj

class SemanticSearchEngine:
    '''
    A semantic search engine for getting right information about right mutual funds

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
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={'normalize_embeddings': config['embedding_model']['normalize_embeddings']})
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

    def retrieve_relevant_documents(self, doc_query: str):
        """
        Retrieves relevant documents based on fund entities and fund attributes in `doc_query`.

        Parameters:
        ----------
        doc_query: User query about details of some fund

        Returns:
        --------
        A list of documents for each fund name extracted with relevant
        fund attributes present
        """
        self.logger.info('Relevant document started')
        entities = extract_fund_entities(doc_query)
        self.logger.info('Entities: %s', entities)

        results = []
        for item in entities:
            fund_name = item['fund_name']

            invalid_keys = {}
            mapped_fund_attribute_keys = ['fund_long_name']
            for fund_attribute in item['fund_attributes']:
                candidate_result, col_relevance_score = self.col_store.similarity_search_with_relevance_scores(fund_attribute['key'], k=1)[0]
                if col_relevance_score < MIN_RELEVANCE_THRESHOLD:
                    invalid_keys[fund_attribute['key']] = 'Data not avaiable'
                else:
                    mapped_fund_attribute_keys.append(candidate_result.page_content)

            retrieved_fund_symbols = []
            try:
                for doc, doc_relevance_score in self.vector_store.similarity_search_with_relevance_scores(fund_name, k=5):
                    self.logger.info('Relevance score for %s is %s ', doc.page_content, doc_relevance_score)
                    if doc_relevance_score >= MIN_RELEVANCE_THRESHOLD:
                        retrieved_fund_symbols.append(doc.metadata['fund_symbol'])
            except AttributeError:
                self.logger.exception('Error retrieving funds')
                return []
            for fund_symbol in retrieved_fund_symbols:
                relevant_fund_info = self.mutual_fund_data.loc[fund_symbol][mapped_fund_attribute_keys].to_dict()
                result_entry = {key: relevant_fund_info.get(key, 'Data not available') for key in mapped_fund_attribute_keys}
                result_entry.update(invalid_keys)
                results.append(relevant_fund_info)
        self.logger.info('Relevant document retrieval finished')
        return results

    def generate_context_aware_response(self, query:str):
        '''
        Returns context-aware response to the query by using
        approapriate knowledge base

        Parameters:
        ----------
        query: User query regarding details of the mutual funds

        Returns:
        -------
        Response from an LLM based on the context provided by the query
        '''
        retrieved_docs = self.retrieve_relevant_documents(query)

        self.logger.info('Retrieval-augmented response generation started')
        context = '\n'.join([label+':'+str(value) for doc in retrieved_docs for label, value in doc.items()])
        prompt = f"""
You are an intelligent and useful semantic search engine that would get the right information about the right funds.
Context: {context}
Now the output should be human-readable text which should be based on the above context only and should be as concise as possible unless specified otherwise.
Do not include any additional info than being asked for.
Provide correct info from context if possible, else respond saying it's not possible without explictly mentioning the context
Input: {query}"""

        # llama_client = LlamaAPI(os.environ['LLAMA_API_KEY'])
        # llm = ChatLlamaAPI(client=llama_client, model='llama3-70b', temperature=0.1)
        llm = ChatOpenAI(model='gpt-4o-mini')
        message = SystemMessage(content=prompt)
        response = llm.invoke([message])
        self.logger.info('Retrieval-augmented response generation completed')
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
        query_context, query_response = semantic_search_engine.generate_context_aware_response(query)
        end_time = time.time()
        print('*****************Query Relevant Documents******************')
        print(query_context)
        print('************************************************')
        print('Your search result is:')
        print(query_response)
        # for query_item in query_response:
        #     print(query_item)
        #     print()
        semantic_search_engine.logger.info('Time taken:%s', end_time-start_time)
