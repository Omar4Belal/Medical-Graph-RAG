import os
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from camel.models.azure_openai_model import AzureOpenAIModel
from camel.types.enums import ModelPlatformType, ModelType
from dataloader import load_high
import argparse
from data_chunk import run_chunk
from utils import *
from camel.models import ModelFactory

def creat_metagraph(args, content, gid, n4j):
    # Azure OpenAI Model setup
    model="gpt-4o-mini"
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
    api_version="2024-08-01-preview"
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME")

    model_config_dict = {
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False  # Set to True if you need streaming responses
    }
    
    # Initialize AzureOpenAIModel
    azure_model = AzureOpenAIModel(
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=model_config_dict,
        api_key=azure_openai_api_key,
        url=azure_endpoint,
        api_version=api_version,
        azure_deployment_name=azure_deployment
    )

    # Set up the KnowledgeGraphAgent with AzureOpenAIModel
    kg_agent = KnowledgeGraphAgent(model=azure_model)
    # Set instance
    uio = UnstructuredIO()
    whole_chunk = content

    if args.grained_chunk == True:
        content = run_chunk(content)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        ans_str = kg_agent.run(element_example, parse_graph_elements=False)
        # print(ans_str)

        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid)
    return n4j

