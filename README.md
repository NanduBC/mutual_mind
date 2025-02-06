# MutualMind
Getting right info from right funds

## Overview
MutualMind is a semantic search engine which can provide right information about the right funds. A user can input mutual fund related queries to the search engine at it will answer the question based on the dataset provided to it. For this project,  `MutualFunds.csv` from Kaggle has been used as the knowledge base.

## Installation
1. Clone this repo to your local machine.
2. Download `MutualFunds.csv` from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/mutual-funds-and-etfs?select=MutualFunds.csv).
3. Install dependencies for this project.
```
$ pip install -r requirements.txt
```
4. Run the data embedding pipeline to ingest the csv file, convert it into `Document` and store them along with the embeddings to a vector store.
```
$ python data_embedding_pipeline.py
```
5. Set LLAMA_API_KEY in environment variable after setting it up in [Llama API console](https://console.llamaapi.com/).
```
$ export LLAMA_API_KEY=<YOUR_API_KEY>
```


