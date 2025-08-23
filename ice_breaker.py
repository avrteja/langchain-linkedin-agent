import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from output_parsers import summary_parser
from third_party.linkedin import scrape_linkedin_profile

# Load environment variables
load_dotenv()

# Main logic
if __name__ == "__main__":
    # Example information
    information = """
    Elon Reeve Musk FRS is a businessman. He is known for his leadership of Tesla, SpaceX, X, and the Department of Government Efficiency. Musk has been considered the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("⚠️ GEMINI_API_KEY not found in .env file")
    # Prompt template — note the use of {information}
    summary_template = """
    Given the information about the person below, please provide:
    1. A short summary.
    2. Two interesting facts about them.

    {format_instructions}

    Information:
    {information}
    """
    linkedin_data = scrape_linkedin_profile("https://www.linkedin.com/in/edenmarco/", False)
    # Create prompt template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": summary_parser.get_format_instructions()
        }
    )

    # Initialize LLM (make sure OPENAI_API_KEY is -set in .env, not COOL_API_KEY unless you override default)
    """
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("COOL_API_KEY")  # Optional: specify custom key
    )
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    # Chain the prompt to the LLM
    chain = summary_prompt_template | llm | summary_parser

    # Run the chain with input
    res = chain.invoke(input={"information": linkedin_data})

    # Print result (res is a BaseMessage object)
    print(res.facts)