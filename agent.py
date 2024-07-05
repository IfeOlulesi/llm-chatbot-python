from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Include the LLM from a previous lesson
from llm import llm


agent_prompt = hub.pull("hwchase17/react-chat")

memory = ConversationBufferWindowMemory(
  memory_key='chat_history',
  k=5,
  return_messages=True,
)

tools = [
  Tool.from_function(
    name="General Chat",
    description="For general chat not covered by other tools",
    func=llm.invoke,
    return_direct=True
  )
]

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
    )

def generate_response(prompt):
  response = agent_executor.invoke({"input": prompt})
  print(response)
  return response['output']
