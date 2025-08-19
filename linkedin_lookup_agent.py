from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain import hub
from tools.tools import get_profile_search
load_dotenv()

def lookup(name: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    template = """ given the full name {name_of_the_person} I want you to get me a link to the linkedin profile page. 
    your answer should only contain a url"""

    prompt_template = PromptTemplate(
        template = template,
        input_variables = ["name_of_the_person"]
    )

    tools_for_the_agent = [
        Tool(
            name = "crawl google 4 linkedin profile page",
            func = get_profile_search,
            description = "useful when you need to get the Linkedin page url"
    )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_the_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_the_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_the_person=name)}
    )

    linkedin_profile_url = result["output"]
    return linkedin_profile_url

if __name__ == "__main__":
    linkedin_url = lookup(name="Eden marco")