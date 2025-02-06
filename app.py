import time
from flask import Flask, render_template, request
from semantic_search_engine import get_semantic_search_engine

app = Flask(__name__)

@app.route('/')
def home():
    '''
    Renders home page of the semantic search engine from which
    user query is collected
    '''
    return render_template('semantic_search.html')

@app.route('/search', methods=['POST'])
def search():
    '''
    Renders search page which displays context aware response
    '''
    query = request.form['question']
    start_time = time.time()
    context, results = search_engine.get_context_aware_response(query)
    end_time = time.time()
    print(context)
    print('Time taken: ', end_time-start_time, 'seconds')
    return render_template('semantic_search_result.html', results=results, question=query)

if __name__ == '__main__':
    search_engine = get_semantic_search_engine()
    # TODO: Set debug to false before submitting assignment
    app.run(debug=True)
