import os
import json
from typing import Optional, List
from openai import OpenAI
from agents.papers import fetch_papers_batch #here we depend on PaperAgent to scrapt the new papers
from agents.agent import Agent

class PaperItem(BaseModel):
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract (original or lightly cleaned)")
    url: str = Field(..., description="Canonical or landing page URL")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score in [0,1] based on abstract vs. user query")

class PaperSelection(BaseModel):
    papers: List[PaperItem]

class ScannerAgent(Agent):
    """
    Scan the latest 200 papers, and based on the semantic similarity between their abstracts and the user’s query, select the top 20 most relevant ones.
    """

    MODEL = "gpt-4o-mini"

    # Rank solely based on semantic similarity of the abstracts, without considering citations
    SYSTEM_PROMPT = """You are a research-scanner agent.
    Your task is to select the top-K academic papers from a provided list that best match the user's request,
    strictly by comparing the user's query against each paper's ABSTRACT text (do not infer from title alone).
    Return only a structured JSON object following the provided schema. Do not include explanations.
    
    Scoring & selection rules:
    - Compute a semantic relevance score in [0, 1] between the user's query and each paper ABSTRACT.
    - Rank by this score (descending) and return exactly K items.
    - If abstracts are missing or empty, exclude those papers.
    - Do NOT hallucinate: only use the given title/abstract/url exactly as provided (light cleanup allowed).
    - Be conservative: if many are weakly related, still choose the top-K but give appropriately lower scores.
    """

    
    USER_PROMPT_PREFIX = """User query:
    {query}
    
    Papers (each with Title, Abstract, URL):
    """
    
    USER_PROMPT_SUFFIX = """
    Return JSON ONLY with this exact schema:
    {
      "papers": [
        {
          "title": "...",
          "abstract": "...",
          "url": "...",
          "score": 0.0
        }
      ]
    }
    Return exactly K items.
    """

    name = "Paper Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        """
        Set up this instance by initializing OpenAI
        """
        self.log("Scanner Agent is initializing")
        self.openai = OpenAI()
        self.log("Scanner Agent is ready")

    def fetch_papers(self, memory, query) -> List[ScrapedDeal]:
        """
        Look up new publised paper of a keyword on RSS feeds
        Return 200 newest related papers
        """
        self.log("Scanner Agent is about to fetch papers from RSS feed")
        
        scraped = Paper.fetch(query)
        result = [scrape for scrape in scraped if scrape.abstract and scrape.citations!=None]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def generate_query(self, user_request: str) -> str:
        """
        Use LLM to transform a free-form user request into a concise search query
        suitable for academic paper retrieval.
        """
        system_prompt = "You are an academic research assistant. Given a user request, produce a concise search query (5-10 words) that can be used to retrieve relevant academic papers. Only output the query."
        result = self.openai.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_request}
            ]
        )
        return result.choices[0].message["content"].strip()

    def make_user_prompt(self, scraped) -> str:
        """
        Create a user prompt for OpenAI based on the scraped deals provided
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List[str]=[], user_request: str) -> Optional[PaperSelection]:
        """
        Call OpenAI to provide a high potential list of deals with good descriptions and prices
        Use StructuredOutputs to ensure it conforms to our specifications
        :param memory: a list of URLs representing deals already raised
        :return: a selection of good deals, or None if there aren't any
        """
        # step1: get the query
        query = generate_query(user_request)

        # step2: search & select the top 20 most related
        scraped = self.fetch_papers(memory, query)
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling OpenAI using Structured Output")
            result = self.openai.beta.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
              ],
                response_format=PaperSelection
            )
            result = result.choices[0].message.parsed
            result.deals = [deal for deal in result.deals if deal.price>0]
            self.log(f"Scanner Agent received {len(result.deals)} selected deals with price>0 from OpenAI")
            return result
        return None
                
