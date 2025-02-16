import os
import time
import yaml

import pandas as pd
import numpy as np
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
MIN_RELEVANCE_THRESHOLD = 0.4
NUM_RETRIEVAL_DOCS = 5
NUM_RETRIEVAL_COLS = 1


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

    def retrieve_similar_columns(self, entity):
        invalid_keys = {}
        mapped_fund_attribute_keys = ['fund_long_name']
        for fund_attribute in entity['fund_attributes']:
            candidate_result, col_relevance_score = self.col_store.similarity_search_with_relevance_scores(
                fund_attribute['key'], k=NUM_RETRIEVAL_COLS)[0]
            if col_relevance_score < MIN_RELEVANCE_THRESHOLD:
                invalid_keys[fund_attribute['key']] = 'Data not avaiable'
            else:
                mapped_fund_attribute_keys.append(candidate_result.page_content)
        return mapped_fund_attribute_keys, invalid_keys

    def retrieve_relevant_documents_dense(self, entities: str):
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
        results = []
        for item in entities:
            fund_name = item['fund_name']

            # Finding relevant documents from Document Vector store
            retrieved_fund_symbols = []
            try:
                for doc, doc_relevance_score in self.vector_store.similarity_search_with_relevance_scores(
                    fund_name, k=NUM_RETRIEVAL_DOCS):
                    self.logger.info('Relevance score for %s is %s ', doc.page_content, doc_relevance_score)
                    if doc_relevance_score >= MIN_RELEVANCE_THRESHOLD:
                        retrieved_fund_symbols.append(doc.metadata['fund_symbol'])
            except AttributeError:
                self.logger.exception('Error retrieving funds')
                return []
            mapped_fund_attribute_keys, invalid_keys = self.retrieve_similar_columns(item)
            for fund_symbol in retrieved_fund_symbols:
                relevant_fund_info = self.mutual_fund_data.loc[fund_symbol][mapped_fund_attribute_keys]\
                    .fillna('Data not available').to_dict()
                result_entry = {}
                for key in mapped_fund_attribute_keys:
                    result_entry[key] = relevant_fund_info.get(key, 'Data not available')
                result_entry.update(invalid_keys)
                results.append(relevant_fund_info)
        self.logger.info('Relevant document retrieval finished')
        return results

    def retrieve_relevant_documents_exact(self, entities):
        results = []
        self.logger.info('Exact Keyword-based retrieval started')
        for entity in entities:
            fund_name = entity['fund_name']
            print('Searching', fund_name, '...')
            mapped_fund_attribute_keys, invalid_keys = self.retrieve_similar_columns(entity)
            relevant_fund_info = self.mutual_fund_data.query(
                f"fund_long_name.str.contains('{fund_name}', na=False, case=False)")[mapped_fund_attribute_keys]\
                .fillna('Data not available').to_dict('records')
            reranked_fund_info = self.rerank_relevant_documents(relevant_fund_info, entity)
            for entity in reranked_fund_info:
                entity.update(invalid_keys)
            results.extend(reranked_fund_info)
        self.logger.info('Exact Keyword-based search finished')
        return results

    def rerank_relevant_documents(self, documents, entity):
        '''
        Rerank documents based on entity semantic similarity
        '''
        self.logger.info("Starting batch-based reranking process")
        query_fund_name = entity["fund_name"]
        query_embedding = np.array(self.embedding_model.embed_query(query_fund_name))
        fund_names = [doc.get("fund_long_name", "") for doc in documents]
        if not fund_names:
            return documents
        doc_embeddings = np.array(self.embedding_model.embed_documents(fund_names))
        # Compute similarity on subset of documents
        similarity_scores = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding))
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_indices = sorted_indices[:NUM_RETRIEVAL_DOCS]
        reranked_docs = [documents[i] for i in sorted_indices]
        self.logger.info("Reranking completed")
        return reranked_docs

    def retrieve_relevant_documents_hybrid(self, doc_query: str):
        self.logger.info('Semantic retrieval started')
        entities = extract_fund_entities(doc_query)
        self.logger.info('Entities: %s', entities)
        self.logger.info("Hybrid retrieval started")
        exact_results = self.retrieve_relevant_documents_exact(entities)
        if len(exact_results) > 0:
            self.logger.info("Exact retrieval returned results")
            return exact_results
        else:
            # TODO: Fall back to sparse search (BM25) and then re-rank using dense vectors
            # instead of just using dense search
            self.logger.info("Exact retrieval returned no results, falling back to semantic search")
            return self.retrieve_relevant_documents_dense(entities)

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
        retrieved_docs = self.retrieve_relevant_documents_hybrid(query)

        self.logger.info('Retrieval-augmented response generation started')
        context = '\n'.join([label+': '+str(value) for doc in retrieved_docs for label, value in doc.items()])
        prompt = f"""
You are MutualMind, an intelligent and useful semantic search engine that would get the right information about the right funds.
Context: {context}
Now the output should be human-readable text which should be based on the above context only and should be as concise as possible unless specified otherwise.
Do not include any additional info than being asked for. Do not answer the query if it's not asking for mutual fund related info that is present in the context.
Provide correct info from context if possible, else respond saying it's not possible without explictly mentioning anything about context.
Input: {query}"""

        # llama_client = LlamaAPI(os.environ['LLAMA_API_KEY'])
        # llm = ChatLlamaAPI(client=llama_client, model='llama3-70b', temperature=0.1)
        llm = ChatOpenAI(model='gpt-4o-mini')
        message = SystemMessage(content=prompt)
        # TODO: Response streaming instead of just invocation
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
        results = semantic_search_engine.retrieve_relevant_documents_hybrid(query)
        end_time = time.time()
        print('*****************Query Relevant Documents******************')
        print('\n'.join([label+': '+str(value) for doc in results for label, value in doc.items()]))
        print('************************************************')
        semantic_search_engine.logger.info('Time taken:%s', end_time-start_time)
