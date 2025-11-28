import os
import json
import logging
from langchain.tools import tool
from tavily import TavilyClient
from playwright.async_api import async_playwright
import asyncio
from urllib.parse import urlparse
import nest_asyncio

# Apply nest_asyncio to allow nested event loops (critical for running async tools in sync contexts)
nest_asyncio.apply()

logger = logging.getLogger(__name__)

# Tavily Setup
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

@tool("tavily_search")
def tavily_search_tool(query: str):
    """Search the web for medical guidelines and articles using Tavily."""
    if not tavily_client:
        return "Error: Tavily API key not set."
    try:
        results = tavily_client.search(query, search_depth="advanced")
        return json.dumps(results.get("results", []), indent=2)
    except Exception as e:
        return f"Search failed: {e}"

@tool("playwright_scrape")
def playwright_scraper_tool(url: str):
    """Scrape content from a medical website using Playwright, handling logins via saved states."""
    
    async def _scrape(target_url: str) -> str:
        domain = urlparse(target_url).netloc
        state_file = f"playwright_states/{domain}.json"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            context_args = {
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            
            if os.path.exists(state_file):
                context_args["storage_state"] = state_file
                logger.info(f"Using saved state for {domain}")
            
            context = await browser.new_context(**context_args)
            page = await context.new_page()
            
            try:
                await page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                text = await page.evaluate("document.body.innerText")
                return text[:8000] 
            except Exception as e:
                return f"Scraping failed: {e}"
            finally:
                await browser.close()

    # Handle Async Loop for Tool Execution
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        return loop.run_until_complete(_scrape(url))
    else:
        return loop.run_until_complete(_scrape(url))

# Export the tools list expecting LangChain/CrewAI compatible tools
tools = [tavily_search_tool, playwright_scraper_tool]
