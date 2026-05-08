from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


model = ChatMistralAI(model="mistral-small-2506")

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

query = "machine learning"

parser = StrOutputParser()

# prompt1 = detailed_prompt.invoke({"topic": query})
# prompt2 = short_prompt.invoke({"topic2": query})



# res1= model.invoke(prompt1)
# response1 = parser.parse(res1)
# print("detailed output: ")
# print(response1.content)

# res2 = model.invoke(prompt2)
# response2 = parser.parse(res2)
# print("short output: ")
# print(response2.content)

# as we can see above to get two outputs or to do two tasks at once we had to write long code instead
# we can use parallel runnables

chain = RunnableParallel({
    "detailed_response": detailed_prompt | model | parser,
    "short_response": short_prompt | model | parser,
})

final_response = chain.invoke({"topic": query})
print(final_response["detailed_response"])
print(final_response["short_response"])