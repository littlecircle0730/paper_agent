from typing import Optional, List
from agents.agent import Agent
# from agents.papers import ScrapedPaper #, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.evaluate_agent import EvaluateAgent
# from agents.messaging_agent import MessagingAgent


class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN
    # THRESHOLD = 50

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.evaluate = EvaluateAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, paper: Paper) -> Opportunity:
        """
        Run the workflow for a particular deal
        :param deal: the deal, summarized from an RSS scrape
        :returns: an opportunity including the discount
        """
        self.log("Planning Agent is evaluating the a potential influence of seleted papers")
        estimate = self.evaluate.evaluate(paper.describe)
        self.log(f"Planning Agent has evaluate the paper with influence as ${estimate:.2f}")
        paper.citations = estimate # update estimated citations to current paper
        return paper

    """
    Make it more flexible 
    """
    def plan(self, memory: List[str] = [], user_request) -> Optional[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EvaluateAgent to estimate them
        3. Use the MessagingAgent to send a notification of deals
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory, user_request)
        if selection:
            rankPapers = [self.run(paper) for paper in selection.papers] #.papers[:5] top 20
            rankPapers.sort(key=lambda rankPapers: rankPapers.citations, reverse=True)
            best = rankPapers[0]
            self.log(f"Planning Agent has identified the most influential paper which has citations ${best.citations:.2f}")
            self.messenger.alert(best)
            self.log("Planning Agent has completed a run")
            return best
        return None