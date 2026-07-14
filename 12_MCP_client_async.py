from dotenv import load_dotenv
from langgraph.graph import StateGraph , START , END
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import asyncio

load_dotenv()

llm1 = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct")
llm = ChatHuggingFace(llm=llm1)



@tool 
def calculator(first_num: float , second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two number.
    supported operations : add , sum , mul , div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {'error':"Division by zero is not allowed"}
            result = first_num/second_num
        else:
            return {"error":f"Unsupported operation '{operation}'"}
        
        return {"first_num":first_num , "second_num":second_num, "operation":operation, "result":result}
    
    except Exception as e:
        return {"error":str(e)}
    



#make tool list 
tools = [calculator]

#make LLM tool aware
llm_with_tools = llm.bind_tools(tools)


#state 
class Chatstate(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]
    
def build_graph():
    
        #graph Node
    async def chat_node(state:Chatstate):
        """LLM node that may answer or request a tool call."""
        messages = state['messages']
        response = await llm_with_tools.ainvoke(messages)
        return {"messages":[response]}
    
    tool_node = ToolNode(tools) #execute tool calls
    
    #graph structure
    graph = StateGraph(Chatstate)
    
    graph.add_node("chat_node",chat_node)
    graph.add_node("tools",tool_node)
    
    graph.add_edge(START,"chat_node")
    
    #If the LLM asked for the tool , go to the ToolNode; else finish 
    graph.add_conditional_edges("chat_node",tools_condition)
    
    graph.add_edge("tools", "chat_node")
    
    chatbot = graph.compile()

    return chatbot

async def main():
    
    chatbot = build_graph()
    #chat requiring tool 
    out = await chatbot.ainvoke({'messages':[HumanMessage(content="FIND THE MODUS OF 1322354 AND 23 AND GIVE ANSWER LIKE A CRICKET COMMENTATOR")]})
    print(out['messages'][-1].content)
    
    
if __name__ == '__main__':
    asyncio.run(main())

    
    
    




