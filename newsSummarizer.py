from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults


model = ChatMistralAI(model="mistral-small-2506")

search_ttool = TavilySearchResults(max_result = 5)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful AI 
    summarize the following news into clear bullet points:
    {news}
    """
)

parser = StrOutputParser()

chain = prompt | model | parser

news_result = search_ttool.run("Latest west bengal election news of 7th may 2026")

result = chain.invoke({
    "news": news_result
})

print(result)


