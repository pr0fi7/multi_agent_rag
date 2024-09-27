import getpass
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# Create Retriever
loader = WebBaseLoader("https://python.langchain.com/docs/expression_language/")
docs = loader.load()
    
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings()
vectorStore = FAISS.from_documents(docs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(
    model='gpt-3.5-turbo-1106',
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search = TavilySearchResults()
retriever_tools = create_retriever_tool(
    retriever,
    "lcel_search",
    "Use this tool when searching for information about Langchain Expression Language (LCEL)."
)
tools = [search, retriever_tools]

# agent = create_openai_functions_agent(
#     llm=model,
#     prompt=prompt,
#     tools=tools
# )

# agentExecutor = AgentExecutor(
#     agent=agent,
#     tools=tools
# )
def create_agent(llm, tools, system_prompt):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages( # we are not using from_template because we want to use chat history and we want to share the agent_scratchpad
            (
                "system",
                "Act as you are the best" + system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), #variable for agent to store state and share it between themselves
        
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    # Ensure the output is a simple string
    output = str(result["output"])
    return {"messages": [HumanMessage(content=output, name=name)]}

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

supervisor_agent = create_agent(model, )


def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        'prompt': system_prompt,
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]

if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agentExecutor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant:", response)