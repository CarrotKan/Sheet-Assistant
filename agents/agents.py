from typing import List
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import os

class SheetAgents:
    def __init__(self, tools: List[Tool]):
        self.tools = tools
        try:
            # Using Together AI with Mixtral model
            self.llm = ChatOpenAI(
                base_url="https://api.together.xyz/v1",
                api_key=os.getenv("TOGETHER_API_KEY"),
                model="meta-llama/Llama-3-70b-chat-hf",
                temperature=0,
                verbose=True
            )
            
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            raise
        
        try:
            # Create a prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are the operator in a RAG system, your role is to pick between the search types you have as tools tom provide users with the answer
                 ALWAYS utilise tools, perpare a plan and follow it, make sure to provide short answers to the user with a response based on the tools you have."""),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create the agent
            self.agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=15
            )
            
        except Exception as e:
            print(f"Error creating agent: {str(e)}")
            raise
    
    def process_query(self, query: str) -> str:
        """Process a query using the agent."""
        try:
            print(f"Processing query: {query}")
            print(f"Available tools: {[tool.name for tool in self.tools]}")
            
            result = self.agent_executor.invoke({"input": query})
            print(f"Agent result: {result}")
            
            return result["output"]
            
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return f"Error processing query: {str(e)}"
    