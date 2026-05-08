from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


model = ChatMistralAI(model="mistral-small-2506")


code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("system", "The code is {topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistan who explains the code in simple words"),
    ("system", "Explain the following code in simple words: {code}")
])

parser = StrOutputParser()

# sequence = code_prompt | model | parser 
# sequence2 = sequence | model | parser

# result1 = sequence.invoke({
#     "topic": "Maximum subarray code in python"
# })

# result2 = sequence.invoke({
#     "topic": result1
# })
# print(result1)
# print("---------- Explanation of the code ----------")
# print(result2)


seq1 = code_prompt | model | parser

seq2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explanation": explain_prompt | model | parser
})

chain = seq1 | seq2

result = chain.invoke({"topic": "code of maximum subarray in python"})

print(result["code"])
print(result["explanation"])