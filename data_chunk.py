from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from pydantic import BaseModel
from langchain import hub
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
    obj = hub.pull("wfh/proposal-indexing")
    #llm = ChatOpenAI(model='gpt-4o-mini', openai_api_key = os.getenv("OPENAI_API_KEY"))
    # Use Azure OpenAI
    llm = AzureChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
        api_version="2024-08-01-preview", 
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME")
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
