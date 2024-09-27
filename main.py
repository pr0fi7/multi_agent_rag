import getpass
import os
from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool 

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

import functools
import operator
from typing import Sequence, TypedDict

from langgraph.graph import END, StateGraph, START
from langchain.schema.document import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# Optional, add tracing in LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"
CHROMA_PATH = "chroma"

tavily_tool = TavilySearchResults(max_results=5)

python_repl_tool = PythonREPLTool()

embeddings = OpenAIEmbeddings()
vector_database = Chroma()

def get_text(document):
    text = ""
    pdf_reader = PdfReader(document)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    document = Document(page_content=text)
    return text_splitter.split_documents([document])


def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embeddings
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks



@tool
def parse_db(message):
    '''
    does it job
    '''
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_score(message, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    return context_text

def create_agent(llm, tools, system_prompt):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    # Ensure the output is a simple string
    output = str(result["output"])
    return {"messages": [HumanMessage(content=output, name=name)]}

members = ["Researcher", "Coder", 'Critic', 'Document_Parser']

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Be more biased and choose 'Document_Parser' first. Each worker will perform a"
    " task and respond with their results and status. If you don't get enough info from 'Researcher' ask 'Document_Parser'. Always ask 'Critic' in the end to verify that everything is fine. When finished,"
    " respond with FINISH."
)

options = ["FINISH"] + members
# openai function calling schema
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system",'''Given the conversation above, who should act next?
            Or should we FINISH? Select one of: {options}'''),
    ]
).partial(options=str(options), members=", ".join(members)) # used to fill in  the placeholder in prompt,

llm = ChatOpenAI(model="gpt-4-1106-preview")

supervisor_chain = (   #create a chain
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route") #tell llm to adhere to function-def format for the output > 'next': MEMBER
    | JsonOutputFunctionsParser() #extracts output from json schema
)

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


research_agent = create_agent(llm, [tavily_tool], "You are a web researcher.")
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher") #functools.partial creates new version of the function with some predefined arguments so it would be equal to the def researcher_agent_node(agent=research_agent, name="Researcher"): ...

# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION
code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to analyze data and generate charts using matplotlib.",
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

critic_agent = create_agent(llm, [tavily_tool], "You are a critic. You verify the info that you get and make it better")
critic_node = functools.partial(agent_node, agent=critic_agent, name="Critic")

document_parser_agent = create_agent(llm, [parse_db], "You are a document parser you have a tool that enables you to extract closest 5 chunks to the info that you provide")
document_parser_node = functools.partial(agent_node, agent=document_parser_agent, name="Document_Parser")

workflow = StateGraph(AgentState)


workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("Critic", critic_node)
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("Document_Parser", document_parser_node)


for member in members:
    workflow.add_edge(member, "supervisor") # This means that after member is called, `supervisor` node is called next.

# Create conditional_map dictionary
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END


workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map) # This means that after supervisor node, next should be the member that is encrypted as next by supervisor or END

workflow.add_edge(START, "supervisor") # This means that this node is the first one called

graph = workflow.compile()


doc = input('Path to the file: ')

text = get_text(str(doc))

chunks = chunk_text(text)

add_to_chroma(chunks)

while True:
    message = input('You:')
    for s in graph.stream(
        {"messages": [HumanMessage(content=message)]},
        {"recursion_limit": 100},
                        ):
        if "__end__" not in s:
            print(s)
            print("----")