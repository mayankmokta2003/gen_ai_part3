from langchain.tools import tool

@tool
def get_greeting(name : str) -> str:
    """ Generates a greeting message for the user """
    return f"<----------> Hello {name}, Welcome to the AI world <---------->"

# now this is a basic custom tool and we gave discription message so that llm knows what this tool does 
# as its also a runnable we can use invoke as well and it also has some methods like:

result = get_greeting.invoke({"name": "Mayank"})
print(result)
print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)


