import time
from flask import Flask, render_template, request
from semantic_search_engine import get_semantic_search_engine

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('semantic_search.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['question']
    start_time = time.time()
    _, results = semantic_search_engine.get_rag_response(query)
    end_time = time.time()
    print('Time taken: ', end_time-start_time, 'seconds')
    return render_template('semantic_search_result.html', results=results, question=query)

if __name__ == '__main__':
    semantic_search_engine = get_semantic_search_engine()
    app.run(debug=True)
