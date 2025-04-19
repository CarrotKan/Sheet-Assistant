from typing import List
from abc import ABC, abstractmethod
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
import os

class BaseSheetAgent(ABC):
    """Base class for sheet agents with different underlying models."""
    
    def __init__(self, tools: List[Tool]):
        self.tools = tools
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.llm = self._initialize_llm()
        self.agent_executor = self._create_agent()
    
    @abstractmethod
    def _initialize_llm(self):
        """Initialize the language model for the agent. Must be implemented by subclasses."""
        pass
    
    def _create_agent(self):
        """Create the agent with the initialized LLM and tools."""
        try:
            # Create a prompt template with memory
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data analyst specialized in analyzing spreadsheet data.
                You're particularly good at understanding context and identifying relevant information.
                Always provide clear, concise answers and explain your findings in a user-friendly way.
                If you find multiple matches or any ambiguity, make sure to mention it.
                If the information seems incomplete, suggest follow-up questions.
                Use the chat history to maintain context and provide more relevant answers."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create the agent
            agent = create_openai_functions_agent(self.llm, self.tools, prompt)
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
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
            
    def get_memory(self) -> List[dict]:
        """Get the conversation history."""
        return self.memory.chat_memory.messages
        
    def clear_memory(self):
        """Clear the conversation history."""
        self.memory.clear() 