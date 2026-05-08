from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from rich import print
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


@tool
def get_text_length(text : str) -> int:
    """ Returns the number of characters in a given text """
    return len(text)

llm = ChatMistralAI(model="mistral-small-2506")

llm_with_tool = llm.bind_tools([get_text_length])

tools = {
    "get_text_length": get_text_length
}

messages = []

query = HumanMessage("Return the number of characters in a given text : 'Mayank Mokta'")
messages.append(query)

result = llm_with_tool.invoke(messages)
messages.append(result)

# print(messages)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    messages.append(tool_message)

final_result = llm_with_tool.invoke(messages)

print(final_result.content)



