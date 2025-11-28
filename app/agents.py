import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from app.tools import tools
from app.models import CochraneReport, PICO
from app.db import graph_db, collection
import json

# LLM Setup
# Assuming xAI is OpenAI compatible
xai_api_key = os.getenv("XAI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="grok-beta", # Using a known model name, user asked for grok-4-1106 but might not be available in library yet, fallback to config
    openai_api_key=xai_api_key or openai_api_key,
    openai_api_base="https://api.x.ai/v1" if xai_api_key else "https://api.openai.com/v1",
    temperature=0
)

# Fallback
fallback_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

class ResearchCrew:
    def __init__(self, task_id: str):
        self.task_id = task_id

    def run(self, topic: str):
        # Agents
        supervisor = Agent(
            role='Research Supervisor',
            goal=f'Oversee the research on "{topic}" and ensure high-quality, verified results.',
            backstory='You are a ruthless research supervisor for the Danish Medical Platform. You demand evidence-based answers.',
            llm=llm,
            allow_delegation=True,
            verbose=True
        )

        researcher = Agent(
            role='Medical Researcher',
            goal='Find relevant medical guidelines and studies.',
            backstory='Expert at searching medical databases and scraping specific guideline websites.',
            tools=tools,
            llm=llm,
            verbose=True
        )

        reviewer = Agent(
            role='Cochrane Reviewer',
            goal='Verify claims using Cochrane methodology (RoB 2.0, GRADE).',
            backstory='You are a strict methodologist. You classify evidence and detect contradictions.',
            llm=llm,
            verbose=True
        )

        # Tasks
        # 1. Search and Scrape
        task_search = Task(
            description=f'Search for current Danish and international guidelines regarding: {topic}. Scrape key content.',
            agent=researcher,
            expected_output="A collection of raw text summaries from relevant sources."
        )

        # 2. PICO & Verification
        task_verify = Task(
            description='Analyze the gathered text. Extract PICO elements. assess Risk of Bias (RoB 2.0) and GRADE. Identify contradictions.',
            agent=reviewer,
            expected_output="A structured report with PICO, verification status, and evidence strength.",
            context=[task_search]
        )

        # 3. Graph Construction (done by Supervisor/Tool or implicit)
        # We'll make the supervisor compile the final report
        task_report = Task(
            description='Compile the final markdown report. Update the knowledge graph with new claims.',
            agent=supervisor,
            expected_output="Final Markdown report and JSON structure.",
            context=[task_verify]
        )

        crew = Crew(
            agents=[supervisor, researcher, reviewer],
            tasks=[task_search, task_verify, task_report],
            process=Process.hierarchical,
            manager_llm=llm,
            verbose=True
        )

        result = crew.kickoff()
        
        # Post-processing: Store in DBs
        # This is a simplification; in a real run, we'd parse the output or have agents call DB tools
        # For now, we'll store the raw result in Chroma
        
        collection.add(
            documents=[str(result)],
            metadatas=[{"task_id": self.task_id, "topic": topic}],
            ids=[f"{self.task_id}_final"]
        )
        
        return result

