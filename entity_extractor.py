import json
import os

from langchain.schema import SystemMessage
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI

from logger import get_logger

logger = get_logger('Entity Extractor')

def extract_fund_entities(query: str) -> dict:
    '''
    Returns extracted fund entities from query. Fund entities include
    fund_name and fund_attributes, where the former is used for
    similarity search on document vector database and the fund
    attribute keys are used to extract the relevant details in
    SemanticSearchEngine
    '''
    prompt = """
You are a Fund Entity Extraction Assistant.
Your job is to analyze a query about mutual funds and extract any fund names, fund attributes (keys and values) mentioned.
Return your answer in JSON with the following keys: "fund_name", "fund_attributes", "key" and "value".
Only "value"s can be absent and in that case set them to null.

Example 1:
Input: "What is the investment strategy of Franklin Income Fund?"
Output:
[
    {
        "fund_name": "Franklin Income Fund",
        "fund_attributes": [
            {
                "key": "investment strategy",
                "value": null
            },
        ]
    }
]

Example 2:
Input: "Sterling funds in large cap growth and their sharpe ratio for 5 years"
Output:
[
    {
        "fund_name": "Sterling funds",
        "fund_attributes": [
            {
                "key": "fund_category",
                "value": "Large"
            },
            {
                "key": "fund_sharpe_ratio_5years",
                "value": null
            }
        ]
    }
]


Example 3:
Input: "Investment strategy of DWS vs Blackrock global funds in large cap"
Output:
[
    {
        "fund_name": "DWS global",
        "fund_attributes": [
            {
            "key": "size_type",
            "value": "Large"
            },
            {
            "key": "investment strategy",
            "value": null
            }
        ]
    },
    {
        "fund_name": "Blackrock global",
        "fund_attributes": [
            {
            "key": "size_type",
            "value": "Large"
            },
            {
            "key": "investment strategy",
            "value": null
            }
        ]
    },
]

Output should contain only JSON that is parseable from Python code.
It should not contain any text at all or other formatting like \\n or `(tilde).
Now, process the following query:
"""
    logger.info('Started entity extraction')
    prompt += f'Input: "{query}"'
    llama_client = LlamaAPI(os.environ['LLAMA_API_KEY'])
    llm = ChatLlamaAPI(client=llama_client, model='llama3-8b', temperature=0)
    
    message = SystemMessage(content=prompt)
    response = llm.invoke([message])
    # print(response.content.replace('\n',''))
    try:
        content = response.content.replace('\n', '').replace('`','')
        extracted = json.loads(content)
    except Exception:
        logger.exception("Error parsing JSON")
        extracted = {}
    logger.info('Entity extraction finished')
    return extracted

if __name__ == '__main__':
    while True:
        print('Entity extractor utility. Type in "Stop" to exit the utility')
        print('Type in your query:', end=' ')
        query = input()
        if query.lower() == 'stop':
            break
        results = extract_fund_entities(query)
        for item in results:
            print('Fund name:', item['fund_name'])
            print('Fund attributes:', item['fund_attributes'])