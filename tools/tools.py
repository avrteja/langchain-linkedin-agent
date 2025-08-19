from langchain_tavily import TavilySearch

def get_profile_search(name: str):
    """searches for linkedin/twitter profile page"""
    search = TavilySearch()
    res = search.run(f"{name}")
    return res