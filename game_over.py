import asyncio
import aiohttp
import os
import json
from typing import List, Dict, Any, Optional, Callable


import faiss
from langchain_community.embeddings import OllamaEmbeddings
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn
from rich.markdown import Markdown
from rich.logging import RichHandler

from loguru import logger

from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor, initialize_agent, Tool, load_tools
from langchain.schema import Document
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_community.vectorstores import FAISS
from atlassian import Confluence
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.tools import BaseTool, StructuredTool
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import justext
from settings import *


console = Console()


class CSVLoader:
    def __init__(self, file_path: str, encoding: str = 'utf8'):
        self.file_path = file_path
        self.encoding = encoding

    async def aload(self) -> List[Document]:
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
        return [Document(page_content=line, metadata={'source': 'TestRail CSV'}) for line in lines[1:]]


async def fetch_confluence_page(session: aiohttp.ClientSession, confluence: Confluence, page_id: str) -> Optional[Document]:
    try:
        content = await asyncio.to_thread(confluence.get_page_by_id, page_id, expand='body.storage')
        logger.info(f"Fetching Confluence page with ID: {page_id}")
        if isinstance(content, dict) and 'body' in content and 'storage' in content['body'] and 'value' in content['body']['storage'] and content['body']['storage']['value']:
            parsed_content = await asyncio.to_thread(
                justext.justext,
                content['body']['storage']['value'],
                justext.get_stoplist("English")
            )
            cleaned_content = "\n".join(
                re.sub(r"[\n\t]+", " ", paragraph.text)
                for paragraph in parsed_content
                if paragraph.text
            )
            return Document(page_content=cleaned_content, metadata={'title': content['title'], 'source': f"Confluence: {content['title']}", 'id': content['id']})
        else:
            logger.error(f"Error loading Confluence page: Unexpected response format: {content}")
            return None
    except Exception as e:
        logger.exception(f"Error fetching Confluence page {page_id}: {str(e)}")
        return None

async def load_confluence_pages(space_key: str, start_page_id: Optional[str] = None) -> List[Document]:
    try:
        confluence = Confluence(
            url=CONFLUENCE_URL,
            username=ALT_USERNAME,
            password=API_TOKEN
        )

        async def fetch_pages(page_ids: List[str]) -> List[Document]:
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_confluence_page(session, confluence, page_id) for page_id in page_ids]
                results = await asyncio.gather(*tasks)
                # filter out None values from the results
                return [doc for doc in results if doc is not None]

        if start_page_id:
            pages_to_fetch = [start_page_id]
            docs = []
            while pages_to_fetch:
                next_pages_to_fetch = []
                current_docs = await fetch_pages(pages_to_fetch)
                logger.info(f"Fetched {len(current_docs)} pages in current iteration.")  # Add this line
                docs.extend(current_docs)
                for doc in current_docs:
                    page = await asyncio.to_thread(confluence.get_page_by_id, doc.metadata['id'], expand='body.storage,children.page')
                    if 'children' in page and 'page' in page['children'] and page['children']['page']['results']:
                        next_pages_to_fetch.extend([child['id'] for child in page['children']['page']['results']])
                pages_to_fetch = next_pages_to_fetch
            return docs
        else:
            pages = await asyncio.to_thread(confluence.get_all_pages_from_space, space=space_key, limit=100)
            return await fetch_pages([page['id'] for page in pages])

    except Exception as e:
        logger.exception(f"Error loading Confluence pages: {str(e)}")
        return []


async def fetch_jira_data() -> List[Document]:
    try:
        jira = JiraAPIWrapper(
            jira_username=ALT_USERNAME,
            jira_api_token=JIRA_API_TOKEN,  # Use the environment variable
            jira_instance_url="https://365retailmarkets.atlassian.net",
            jira_cloud=True
        )
        jql_query = 'PROJECT = "SOS Portfolio" AND "Team[Team]" = 4b0df4d1-98e6-4661-8eea-5a8d4e5a721a ORDER BY created DESC'
        jql_result = await asyncio.to_thread(jira.search, jql_query)
        logger.info(f"Fetched {len(jql_result)} Jira issues")
        return [Document(page_content=str(x), metadata={'source': 'Jira Search Results'}) for x in jql_result]
    except Exception as e:
        logger.exception(f"Error fetching Jira data: {str(e)}")
        return []

async def load_testrail_data(file_path: str) -> List[Document]:
    loader = CSVLoader(file_path, encoding='utf8')
    return await loader.aload()



async def qa_tool(input: Dict[str, Any], qa_chain: RetrievalQAWithSourcesChain) -> str:
    # Extract the question from the action_input dictionary
    question = input.get("kwargs", {}).get("title", "") 

    if not question:
        return "Error: No question provided"

    result = await qa_chain.ainvoke({"question": question})
    return f"Answer: {result['answer']}\nSources: {result['sources']}"

async def testrail_tool(input: Dict[str, Any], testrail_chain: RetrievalQAWithSourcesChain) -> str:
    if testrail_csv_path:
        testrail_docs = await load_testrail_data(testrail_csv_path)
    
    question = input.get("kwargs", {}).get("title", "")
    if not question:
        return "Error: No question provided for TestRail"
    
    result = await testrail_chain.ainvoke({"question": question, "context": testrail_docs}) 
    return f"TestRail Answer: {result['answer']}\nSources: {result['sources']}"


async def generate_test_strategies_tool(query: str, vectorstore: FAISS, llm: Any, existing_strategy: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.9})
    docs = await retriever.ainvoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_template = ChatPromptTemplate.from_template(
        """
        Append to the following test strategy document based on this query: {query}
        
        Use this context to inform the addition:
        {context}

        Existing Test Strategy:
        {existing_strategy}
        
        Format the output using Markdown. Only include the updated test strategy section, not the full document.
        """
    )

    chain = prompt_template | llm | StrOutputParser()
    addition = await chain.ainvoke(dict(query=query, context=context, existing_strategy=existing_strategy))


    existing_strategy += addition 
    
    return addition

async def run_agent(max_retries=3, testrail_csv_path: Optional[str] = None):
    logger.info("Starting agent run...")
    
    confluence_docs_task = asyncio.create_task(load_confluence_pages(space_key="QA", start_page_id="2862907405")) 
    jira_data_task = asyncio.create_task(fetch_jira_data())
    
    confluence_docs, jira_docs = await asyncio.gather(confluence_docs_task, jira_data_task)
    
    all_docs = confluence_docs + jira_docs 

    logger.info(f"Total number of documents: {len(all_docs)}")
    if not all_docs:
        logger.warning("No documents loaded. Ensure Confluence and Jira configurations are correct.")
        return {"error": "No documents loaded"}
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)
    logger.info(f"Split documents into {len(split_docs)} chunks.")

    # Use concurrent.futures for parallel embedding
    from concurrent.futures import ThreadPoolExecutor

   
    def embed_chunk(chunk: Document) -> Any:
        return (chunk, cached_embedder.embed_documents([chunk.page_content])[0])  # Return tuple of chunk and embedding

    with ThreadPoolExecutor() as executor:
        text_embeddings = list(executor.map(embed_chunk, split_docs))

    # Create the FAISS index with the embedded documents
    logger.info(f"Creating FAISS index with {len(text_embeddings)} embeddings.")
    vectorstore = FAISS.from_embeddings([(chunk.page_content, embedding) for chunk, embedding in text_embeddings], cached_embedder)


    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Initialize the RetrievalQAWithSourcesChain with the GPTCache object 
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(Ollama(base_url="http://localhost:13000"), retriever=retriever)

    if testrail_csv_path:
        testrail_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        # Initialize the RetrievalQAWithSourcesChain with the GPTCache object
        testrail_chain = RetrievalQAWithSourcesChain.from_chain_type(Ollama(base_url="http://localhost:13000"), retriever=testrail_retriever)

    existing_strategy = ""

    def _qa_tool_wrapper(**kwargs):
        return qa_tool(kwargs, qa_chain)

    def _testrail_tool_wrapper(**kwargs):
        return testrail_tool(kwargs, testrail_chain)

    def _generate_test_strategies_tool_wrapper(query, existing_strategy):
        return generate_test_strategies_tool(query, vectorstore, Ollama(base_url="http://localhost:13000"), existing_strategy)

    def create_qa_tool() -> StructuredTool:
        return StructuredTool.from_function(
            func=_qa_tool_wrapper,
            name="QA System",
            description="Useful for answering questions about the Confluence and Jira documents. Provide the question in the 'title' field of the 'kwargs' dictionary.",
            args=[],
            kwargs={"kwargs": {"title": ""}},
        )

    def create_testrail_tool() -> StructuredTool:
        return StructuredTool.from_function(
            func=_testrail_tool_wrapper,
            name="TestRail System",
            description="Useful for answering questions about the TestRail CSV document. Provide the question in the 'title' field of the 'kwargs' dictionary.",
            args=[],
            kwargs={"kwargs": {"title": ""}},
        )

    def create_generate_test_strategies_tool() -> StructuredTool:
        return StructuredTool.from_function(
            func=_generate_test_strategies_tool_wrapper,
            name="Generate Test Strategies",
            description="Generates test strategies for a given functionality or issue. Provide a concise description of the functionality or the issue key. Output will be appended to the existing test strategy document.",
            args=["query", "existing_strategy"],
            kwargs={},
        )


    tools = [
        create_qa_tool(),
        create_generate_test_strategies_tool()
    ]
    if testrail_csv_path:
        tools.append(create_testrail_tool())

    tool_names = [tool.name for tool in tools]

    llm = Ollama(base_url="http://localhost:13000")

    initial_prompt = PromptTemplate(
        template="""
        **Context:** {context}
        **Tools:** {tools}
        **Tool Names:** {tool_names}
        **Question:** {question}

        Here are the key sections the final test strategy document should include:

        1. Objective: Clearly state the goals and objectives of testing the SOS mobile application.
        2. Scope: Define what's in scope and out of scope for testing. Focus on key mobile functionality, risks, and quality metrics.  
        3. Test Approach: Outline the overall approach to testing, including types of testing (functional, usability, performance, security, etc.), test levels, and techniques.
        4. Test Environment: Specify the hardware, software, tools, and configurations required for testing.
        5. Test Data: Describe the test data requirements and how test data will be prepared. 
        6. Test Execution Schedule: Provide a high-level schedule for test activities.
        7. Roles & Responsibilities: Identify the key roles and their responsibilities for the testing effort.
        8. Risks & Mitigation: Highlight the risks associated with the testing and describe the mitigation strategy for each.
        9. Test Deliverables: List the documents, reports and other artifacts that will be delivered as part of testing.

        **Thought:** I will use the QA tool to gather relevant context about the SOS mobile app and key areas to focus on. Then I'll use the Generate Test Strategies tool to incrementally build out each section of the test strategy document, using the context to inform the content.
        Use the TestRail System to inform the Test Execution Schedule and test coverage.

        {agent_scratchpad}
        """,
        input_variables=["context", "tools", "tool_names", "question", "agent_scratchpad"]
    )


    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, prompt=initial_prompt)

    logger.info("Running the agent...")
    console.print(Panel(f"Querying agent with question:\n{QUESTION}", title="Query"))

    for attempt in range(max_retries):
        logger.info(f"Running agent, attempt {attempt + 1} of {max_retries}")
        try:
            response = await agent.ainvoke({
                "input": QUESTION,
                "question": QUESTION,
                "context": "The actual context will be retrieved by the QA System tool.",
                "tools": tools,
                "tool_names": tool_names,
                "agent_scratchpad": ""
            })


            # Collect the generated test strategy
            for tool_call in response['intermediate_steps']:
                if "Generate Test Strategies" in tool_call:
                    existing_strategy += tool_call.split("```tool_code\n")[1].split("```")[0]
                    break
            else:
                logger.warning("Generate Test Strategies tool was not called. Unable to retrieve test strategy.")

            test_strategy_doc = Markdown(existing_strategy)
            console.print(Panel(test_strategy_doc, title="Test Strategy Document"))
            result = {
                "test_strategy_document": existing_strategy,
                "agent_output": response['output']
            }

            logger.info(json.dumps(result, indent=2))
            logger.success("Agent run completed successfully")

            # Save results to disk
            with open("agent_results.json", "w") as f:
                json.dump(result, f, indent=2)
            logger.info("Results saved to agent_results.json")

            return result

        except Exception as e:
            logger.error(f"An error occurred during agent execution: {str(e)}")
            if attempt < max_retries - 1:
                logger.warning("Retrying...")
            else:
                return {"error": str(e)}

    logger.error(f"Agent failed after {max_retries} attempts.")
    return {"error": "Agent failed to find relevant information"}


if __name__ == "__main__":
    testrail_csv_path = ""
    asyncio.run(run_agent(testrail_csv_path=testrail_csv_path))
