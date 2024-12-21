from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains.openai_functions import create_extraction_chain_pydantic
from pydantic import BaseModel
from langchain import hub # The `langchainhub sdk` is deprecated. Please use the `langsmith sdk` instead
# from langsmith import hub
from langsmith import Client
import os
from dataloader import load_high
from agentic_chunker import AgenticChunker

# Pydantic data class
class Sentences(BaseModel):
    sentences: List[str]


def get_propositions(text, runnable, extraction_chain):
    runnable_output = runnable.invoke({
    	"input": text
    }).content
    
    # processes the output to extract propositions (sentences validated by Sentences schema
    propositions = extraction_chain.run(runnable_output)[0].sentences
    return propositions

def run_chunk(essay):

    # pulls a pre-defined LangChain prompt template named "wfh/proposal-indexing"
    client = Client()
    obj = client.pull_prompt("wfh/proposal-indexing")
    #llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key = os.getenv("OPENAI_API_KEY"))
    # Use Azure OpenAI
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    llm = AzureChatOpenAI(
            model="gpt-4o-mini", 
            api_key=azure_openai_api_key,
            api_version="2024-08-01-preview",
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment
    )

    runnable = obj | llm

    # Extraction
    extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm) 

    paragraphs = essay.split("\n\n")

    essay_propositions = []

    for i, para in enumerate(paragraphs):
        propositions = get_propositions(para, runnable, extraction_chain)
        
        essay_propositions.extend(propositions)
        print (f"Done with {i}")

    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='list_of_strings')

    return chunks
    print(chunks)

