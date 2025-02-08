# [MutualMind](https://github.com/NanduBC/mutual_mind)
Getting right info from right mutual funds

## Overview
MutualMind is a semantic search engine which can provide right information about the right funds. A user can input mutual fund related queries to the search engine at it will answer the question based on the dataset provided to it. For this project,  `MutualFunds.csv` from Kaggle has been used as the knowledge base.

## Installation
1. Clone this repo to your local machine.
2. Download `MutualFunds.csv` from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/mutual-funds-and-etfs?select=MutualFunds.csv).
3. Install dependencies for this project.
```
$ pip install -r requirements.txt
```
4. Execute the data embedding pipeline to ingest the csv file, convert it into `Document` and store them along with the embeddings to a vector store.
```
$ python data_embedding_pipeline.py
```
5. Set `OPENAI_API_KEY` in environment variable after creating the token in [Open AI account](https://platform.openai.com/api-keys)
```
$ export OPENAI_API_KEY=<YOUR_API_KEY>
```

## Running Semantic Search Engine

There are two ways of interaction with the semantic search engine. A Flask-based web interface has been developed with minimal set of features where user can input a query and then search for relevent information about the funds. Same set of queries can be run on command line as well.

### 1. Querying through Web interface
Run the following script to get the Flask app up and running. And paste the url on a browser to test the app
```
$ python app.py
```

### 2. Querying through command line
Alternatively user can type in query through command line after executing the following script
```
$ python semantic_search_engine.py
```
To stop the execution, one can type `Stop`


