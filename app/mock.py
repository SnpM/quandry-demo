from quandry.classes import *

country_capitals = {
    "France":"Paris",
    "Germany":"Berlin",
    "Italy":"Rome",
    "Spain":"Madrid",
    "Portugal":"Lisbon",
}
class HumanSubject(ISubject):
    responses = {}
    def update(self,responses:dict):
        self.responses.update(responses)
    def respond(self, prompt: str) -> str:
        if prompt in self.responses:
            return self.responses[prompt]
        else:
            return "Error: No answer to prompt."

class CapitalTriviaSubject(ISubject):
    def respond(self, prompt: str) -> str:
        for country,capital in country_capitals.items():
            if country.lower() in prompt.lower():
                return capital
        return "I don't know"
    
class CapitalTriviaEvaluator(IEvaluator):
    def evaluate(self, prompt:str, expectation:str, response:str) -> Evaluation:
        correct_answer = None
        for country,capital in country_capitals.items():
            if (country.lower() in prompt.lower()) or (capital.lower() in expectation.lower()):
                correct_answer = capital

        if correct_answer is None:
            return Evaluation(EvalCode.ERROR, f"I don't know how to evaluate the prompt")
        
        if correct_answer.lower() in response.lower():
            return Evaluation(EvalCode.PASS, f"Correctly responded with '{correct_answer}'")
        else:
            return Evaluation(EvalCode.FAIL, f"Failed to include '{correct_answer}' in response to prompt.")