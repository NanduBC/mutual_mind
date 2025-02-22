import yaml

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from logger import get_logger


with open('embedding_model_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

EMBEDDING_MODEL_NAME = config['embedding_model']['name']
logger = get_logger('Test Similarity Search')
logger.info('Loading embedding model:%s', EMBEDDING_MODEL_NAME)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={'normalize_embeddings': config['embedding_model']['normalize_embeddings']})

logger.info('Loading vector store')
vector_store = Chroma(
    persist_directory="./mutual_funds_store",
    embedding_function=embedding_model,
    collection_name="funds-categorical")
logger.info('Vector Store loaded with #docs:%s', vector_store._collection.count())

print('Semantic Search Utility. Type "Stop" to stop utility')
while True:
    print('Type in your query:', end=' ')
    query = input()
    if query.lower() == 'stop':
        break
    # Perform a similarity search
    query = config['embedding_model']['query_prefix']
    results = vector_store.similarity_search_with_relevance_scores(query, k=3)
    print('********************************************')
    for doc, relevance_score in results:
        print('Document content:', doc.page_content)
        print('Document metadat:', doc.metadata)
        print('Document relevance score:', relevance_score)
        print()
    print('********************************************')
