import time
from flask import Flask, render_template, request

from logger import get_logger
from semantic_search_engine import get_semantic_search_engine


app = Flask(__name__)
logger = get_logger('MutualMindWebServer')

@app.route('/')
def home():
    '''
    Renders home page of the semantic search engine from which
    user query is collected
    '''
    return render_template('semantic_search.html')

@app.route('/core_search', methods=['POST'])
def search():
    '''
    Renders search page which displays relevant information based
    on the query
    '''
    query = request.form['question']
    start_time = time.time()
    results = search_engine.retrieve_relevant_documents(query)
    end_time = time.time()
    logger.info('Time taken: %s seconds', end_time-start_time)
    return render_template('semantic_search_result.html', results=results, question=query)

@app.route('/ai_search', methods=['POST'])
def ai_search():
    '''
    Renders search page which displays context-aware relevant information
    based on the query
    '''
    query = request.form['question']
    start_time = time.time()
    context, results = search_engine.generate_context_aware_response(query)
    end_time = time.time()
    logger.info(context)
    logger.info('Time taken: %s seconds', end_time-start_time)
    return render_template('semantic_search_advanced_result.html', results=results, question=query)

if __name__ == '__main__':
    search_engine = get_semantic_search_engine()
    # TODO: Set debug to false before submitting assignment
    app.run(debug=True)
