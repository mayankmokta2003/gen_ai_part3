from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda



# its used when topic are of same name in both tempelates but we have to give different input in each

model = ChatMistralAI(model="mistral-small-2506")

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

parser = StrOutputParser()

chain = RunnableParallel({
    "detailed": RunnableLambda(lambda x: x["detailed"]) | detailed_prompt | model | parser,
    "short": RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser
})

response = chain.invoke({
    "detailed": "machine learning",
    "short": "deep learning"
})

print(response["detailed"])
print(response["short"])