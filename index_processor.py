import os
import json

import pandas as pd
import numpy as np
from tqdm import tqdm
# from llamaapi import LlamaAPI
from langchain.schema import Document
# from langchain_experimental.llms import ChatLlamaAPI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema import SystemMessage


def create_documents_from_dataframe(fund_data):
    docs = []
    for item in fund_data.to_dict('records'):
        attributes = []
        relevant_keys = ['fund_long_name']# 'fund_symbol', 'fund_category', 'fund_family', 'management_name']
        metadata_keys = ['fund_symbol', 'investment_type', 'size_type', 'fund_category', 'fund_family', 'management_name']
        metadata = dict()
        for key,value in item.items():
            if key in relevant_keys:      
                attributes.append(str(value))
            if key in metadata_keys:
                metadata[key] = value
        doc_content = '\n'.join(attributes)
        docs.append(Document(page_content=doc_content, metadata=metadata))
    return docs

def create_vector_store(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="funds-categorical",
        embedding_function=embedding_model,
        persist_directory='./mutual_funds_store'
        )
    
    ids_to_delete = vector_store.get()['ids']
    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)
    batch_size = 1000
    num_batches = len(documents) // batch_size + (1 if len(documents) % batch_size > 0 else 0)

    for iteration in tqdm(range(num_batches), desc="Adding documents"):
        start_index = iteration * batch_size
        end_index = min(start_index + batch_size, len(documents))
        doc_ids = [doc.metadata['fund_symbol'] for doc in documents[start_index:end_index]]
        vector_store.add_documents(documents[start_index:end_index])
    print('Vector store lenght:', len(vector_store.get()['ids']))
    return vector_store

if __name__ == '__main__':
    mutual_funds = pd.read_csv('MutualFunds.csv')
    print('Dataset shape:', mutual_funds.shape)
    categorical_features = mutual_funds.select_dtypes(include='object')
    numeric_features = mutual_funds.select_dtypes(include='float64')

    documents = create_documents_from_dataframe(categorical_features)
    vector_store = create_vector_store(documents)

    # results = vector_store.similarity_search("Franklin", k=5)
    # for doc in results:
    #     print(doc.page_content, doc.metadata)
