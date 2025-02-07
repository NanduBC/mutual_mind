import yaml

import pandas as pd
from tqdm import tqdm
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from logger import get_logger

with open('embedding_model_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

EMBEDDING_MODEL_NAME = config['embedding_model']['name']
logger = get_logger('Data Embedding Pipeline')
logger.info('Loading embedding model:%s', EMBEDDING_MODEL_NAME)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={'normalize_embeddings': config['embedding_model']['normalize_embeddings']})


def create_documents_from_dataframe(fund_data: pd.DataFrame):
    '''
    Creates list of `Document`s from dataframe

    Parameters:
    -----------
    fund_data: A pandas DataFrame containig details of mutual fund

    Returns:
    --------
    List of `Document`s containing identifiable features for a fund in
    the content and some useful fields in metadata
    '''
    logger.info('Document creation from dataframe started')
    docs = []
    for item in fund_data.to_dict('records'):
        attributes = []
        relevant_keys = ['fund_long_name', 'fund_short_name', 'fund_family', 'management_name']
        metadata_keys = ['fund_symbol', 'investment_type', 'size_type', 'fund_category', 'fund_family', 'management_name']
        metadata = dict()
        for key,value in item.items():
            if key in relevant_keys:      
                attributes.append(str(value))
            if key in metadata_keys:
                metadata[key] = value
        doc_content = '\n'.join(attributes)
        doc_content = config['embedding_model']['passage_prefix'] + doc_content
        docs.append(Document(page_content=doc_content, metadata=metadata))
    logger.info('Document creation completed')
    return docs

def create_doc_vector_store(documents:list[Document]):
    '''
    Creates a vector store with fund-related Documents

    Parameters:
    ----------
    documents: List of `Document`s containing relevant fund fields
     for similarity search
    '''
    logger.info('Vector store creation started')
    doc_vector_store = Chroma(
        collection_name='funds-categorical',
        embedding_function=embedding_model,
        persist_directory='./mutual_funds_store')
    
    ids_to_delete = doc_vector_store.get()['ids']
    if ids_to_delete:
        doc_vector_store.delete(ids=ids_to_delete)
    batch_size = 1000
    num_batches = len(documents) // batch_size + (1 if len(documents) % batch_size > 0 else 0)

    for iteration in tqdm(range(num_batches), desc='Adding documents'):
        start_index = iteration * batch_size
        end_index = min(start_index + batch_size, len(documents))
        doc_vector_store.add_documents(documents[start_index:end_index])
    logger.info('Vector store lenght:%s', len(doc_vector_store.get()['ids']))
    logger.info('Vector store creation completed')

def create_col_vector_store(fund_data:pd.DataFrame):
    '''
    Creates a vector store with Columns as documents

    Parameters:
    ----------
    mutual_funds: Mutual funds dataframe
    '''
    col_vector_store = Chroma(
        collection_name='cols-store',
        embedding_function=embedding_model,
        persist_directory='./mf_cols'
        )

    col_ids_to_delete = col_vector_store.get()['ids']
    if col_ids_to_delete:
        col_vector_store.delete(ids=col_ids_to_delete)

    column_as_doc = []
    for col in fund_data.columns.to_list():
        column_as_doc.append(Document(page_content=col))
    col_vector_store.add_documents(column_as_doc)

if __name__ == '__main__':
    mutual_funds = pd.read_csv('MutualFunds.csv')
    print('Dataset shape:', mutual_funds.shape)
    categorical_features = mutual_funds.select_dtypes(include='object')
    numeric_features = mutual_funds.select_dtypes(include='float64')

    documents = create_documents_from_dataframe(categorical_features)
    create_doc_vector_store(documents)
    create_col_vector_store(mutual_funds)
