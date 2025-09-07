from typing import Optional, List
from PaperAgent.agents.agent import Agent
from PaperAgent.agents.scanner_agent import ScannerAgent
from PaperAgent.agents.evaluate_agent import EvaluateAgent
from PaperAgent.agents.messaging_agent import MessagingAgent
from PaperAgent.agents.papers import Paper

class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()
        # self.evaluate = EvaluateAgent(collection) ################### TODO:!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.messenger = MessagingAgent()        ################### TODO:!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.log("Planning Agent is ready")

    def run(self, paper: Paper) -> Paper:
        """
        Run the workflow for a particular paper
        :paper: the paper, summarized from an RSS scrape
        :returns: an Paper including the citation
        """
        self.log("Planning Agent is evaluating the a potential influence of seleted papers")
        
        # # estimate = self.evaluate.evaluate(paper.describe) ###### TODO:!!!!!!!!!!!!!!!!!
        # self.log(f"Planning Agent has evaluate the paper with influence as ${estimate:.2f}")
        # paper.citations = estimate # update estimated citations to current paper
        
        return paper

    """
    Make it more flexible ???
    """
    def plan(self, memory: List[str] = [], user_request="Give me 5 most recent papers in CS") -> Optional[Paper]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EvaluateAgent to estimate them
        3. Use the MessagingAgent to send a notification of deals
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=None, user_request=user_request)
        if selection:
            rankPapers = [self.run(paper) for paper in selection.papers] #.papers[:5] top 20
            rankPapers.sort(key=lambda rankPapers: rankPapers.citations, reverse=True) #TODO: change it to use model to predict
            best = rankPapers[:5]
            self.log(f"Planning Agent has identified the most influential paper which has citations {[f'{ele.citations:.2f}' for ele in best]}")
            # self.messenger.alert(best) #TODO:message agent
            self.log("Planning Agent has completed a run")
            return best
        return None

# test
# RUN: python -m PaperAgent.agents.scanner_agent
if __name__ == "__main__":
    query = "I would like to know what is the tendency of tech in the next 10 years"
    collection = []
    planner = PlanningAgent(collection)
    selectedPapers = planner.plan(memory=None, user_request=query)
    if selectedPapers:
        for i, paper in enumerate(selectedPapers):
            print("### selected Papers: /n", paper.describe())
            print("\n")