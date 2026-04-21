import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Selecting the minimum length description of the available data as the best model observes the principle identified as Occam's razor. Prior to the advent of computer programming, generating such descriptions was the intellectual labor of scientific theorists. It was far less formal than it has become in the computer age. If two scientists had a theoretic disagreement, they rarely could formally apply Occam's razor to choose between their theories. They would have different data sets and possibly different descriptive languages. Nevertheless, science advanced as Occam's razor was an informal guide in deciding which model was best.
    """
    summary_template = f"""
    Given the information {information} about the model i want you yo create:
    1. A short summary
    2. keep it concise and to the point
    """
    summary_prompt = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    summary_chain = summary_prompt | llm
    summary = summary_chain.invoke({"information": information})
    print(summary.content)


if __name__ == "__main__":
    main()
