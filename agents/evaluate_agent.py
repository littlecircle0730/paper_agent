import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

from PaperAgent.agents.agent import Agent
from PaperAgent.agents.specialist_agent import SpecialistAgent
from PaperAgent.agents.frontier_agent import FrontierAgent
from PaperAgent.agents.random_forest_agent import RandomForestAgent

class EvaluateAgent(Agent):

    name = "Ensemble Agent"
    color = Agent.YELLOW
    
    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        # self.random_forest = RandomForestAgent() ###########################################
        # self.model = joblib.load('ensemble_model.pkl') ##############################################
        self.log("Ensemble Agent is ready")

    def evaluate(self, description: str) -> float:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - collaborating with specialist, frontier and random forest agents")
        specialist = self.specialist.price(description)
        frontier = self.frontier.price(description)
        random_forest = self.random_forest.price(description)
        X = pd.DataFrame({
            'Specialist': [specialist],
            'Frontier': [frontier],
            'RandomForest': [random_forest],
            'Min': [min(specialist, frontier, random_forest)],
            'Max': [max(specialist, frontier, random_forest)],
        })
        y = max(0, self.model.predict(X)[0])
        self.log(f"Ensemble Agent complete - returning ${y:.2f}")
        return y