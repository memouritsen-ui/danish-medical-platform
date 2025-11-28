import os
import json
import logging
from typing import Optional, Type
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from tavily import TavilyClient
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Tavily Setup
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

class SearchSchema(BaseModel):
    query: str = Field(description="The search query to run.")

class ScrapeSchema(BaseModel):
    url: str = Field(description="The URL to scrape.")

# Using the @tool decorator is often safer for CrewAI compatibility as it handles the wrapping correctly
# However, CrewAI 0.22+ often expects instances of BaseTool subclasses directly, but Pydantic v2 validation can be tricky.
# Let's try to fix the Pydantic error by ensuring strict typing or using the decorator approach which is cleaner.

@tool("tavily_search")
def tavily_search_tool(query: str):
    """Search the web for medical guidelines and articles."""
    if not tavily_client:
        return "Error: Tavily API key not set."
    try:
        results = tavily_client.search(query, search_depth="advanced")
        return json.dumps(results.get("results", []), indent=2)
    except Exception as e:
        return f"Search failed: {e}"

@tool("playwright_scrape")
def playwright_scraper_tool(url: str):
    """Scrape content from a medical website, handling logins via saved states."""
    
    async def _scrape(url: str) -> str:
        domain = urlparse(url).netloc
        state_file = f"playwright_states/{domain}.json"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            # Context with state
            context_args = {"user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            if os.path.exists(state_file):
                context_args["storage_state"] = state_file
                logger.info(f"Using saved state for {domain}")
            
            context = await browser.new_context(**context_args)
            page = await context.new_page()
            
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                text = await page.evaluate("document.body.innerText")
                return text[:8000] # Limit context
            except Exception as e:
                return f"Scraping failed: {e}"
            finally:
                await browser.close()

    # Run async in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
         # If we are already in an async loop (e.g. FastAPI), we can't nest run_until_complete
         # This is a common pain point. For CrewAI which is often sync, this usually works.
         # But if Crew calls this from inside an async context, it fails.
         # We'll try a nest_asyncio patch if needed, but first let's try proper isolation.
         import nest_asyncio
         nest_asyncio.apply()
    
    return loop.run_until_complete(_scrape(url))

# Export the decorated functions which are now LangChain tools
tools = [tavily_search_tool, playwright_scraper_tool]
