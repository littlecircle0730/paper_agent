from typing import Dict, List, Optional
from pydantic import BaseModel
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time


class Paper(BaseModel):
    title: str
    citations: int
    abstract: str = ""
    url: str
    authors: List[str] = []
    published: Optional[str] = None
    paper_id: Optional[str] = None
    version: Optional[str] = None

    def describe(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Citations: {self.citations}\n"
            f"Abstract: {self.abstract}\n"
            # f"Authors: {', '.join(self.authors) if self.authors else 'N/A'}\n"
            f"Published: {self.published or 'N/A'}\n"
            f"paper ID: {self.paper_id or 'N/A'} {self.version or ''}\n"
            f"URL: {self.url}\n"
        )
        
    def make_model_input(self) -> str:
        year = getattr(self, "year", None) or _extract_year(getattr(self, "published", None)) or ""
        abstract = (self.abstract or "").strip()

        return (
            f"Year: {year}\n"
            f"Abstract: {abstract}\n"
            f"Number of citations: "
        )

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "citationCount": self.citations,
            "url": self.url,
            "published": self.published,
            "paperId": self.paper_id,
            "version": self.version,
            "authors": self.authors,
        }

    @classmethod
    def fetch_papers_batch(cls, query: str, start_offset: int = 0, limit: int = 100, year="-2025", timeout: int = 20) -> List[Dict]:
        offset = start_offset * limit
        
        def _headers() -> Dict[str, str]:
            return {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "paper-citation-checker/1.0",
                # "x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # Optional API key if needed
            }

        # TODO: use offset to get training 1000 data
        BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "fields": "title,year,url,externalIds,citationCount,abstract",
            "offset": offset,
            "limit": limit,
            "bulk": "true",
            "sort": "publicationDate:desc",
            "year": year,
        }
    
        try:
            print("Waiting for paper collection")
            response = requests.get(BASE_URL, params=params, headers=_headers(), timeout=timeout)
            if response.status_code == 429:
                print("[WARN] Rate limited. Retrying...")
                time.sleep(30.0)
                response = requests.get(BASE_URL, params=params, headers=_headers(), timeout=timeout)
    
            if not response.ok:
                print(f"[ERROR] API request failed: {response.status_code}")
                return []
    
            data = response.json()
            # for d in data.get("data", []):
                # if d.get("abstract"):
                #     print("OK")
                # else:
                #     print("NO")
            return data.get("data", []) or []
    
        except Exception as e:
            print(f"[ERROR] Exception occurred: {e}")
            return []

    @classmethod    
    def fetch(cls, query:str, start_offset: int = 0, limit:int=100, year="2024-2024") -> List["Paper"]:
        papers = []
        scrapedPapers = cls.fetch_papers_batch(query, start_offset, limit, year)
        for p in scrapedPapers:
            papers.append(
                Paper(
                    title=p.get("title", ""),
                    abstract=p.get("abstract") or "",
                    citations=p.get("citationCount", -1),
                    url=p.get("url", ""),
                    published=str(p.get("year", "")) if p.get("year") else None,
                    paper_id=p.get("paperId", ""),
                )
            )
        return papers


# test
if __name__ == "__main__":
    # scraped = ScrapedPaper.fetch(limit=5)
    query = "Generative AI"
    papers = Paper.fetch(query, limit=20)
    for i, sp in enumerate(papers, 1):
        print(f"--- #{i} ---")
        print(sp.describe())