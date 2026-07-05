from pydantic import BaseModel, Field
from typing import List
from typing import TypedDict, Optional, List
from ddgs import DDGS
import numexpr
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

class RiskProfile(BaseModel):
    risk_category: str = Field(description="conservative, moderate, or aggressive")
    reasoning: str

class FundCandidate(BaseModel):
    fund_name: str
    source_url: str
    reason: str = Field(description="why this fund fits the risk profile")

class FundList(BaseModel):
    funds: List[FundCandidate]

class InvestmentPlan(BaseModel):
    monthly_investment: float
    expected_annual_return_pct: float
    reasoning: str

class FinalRecommendation(BaseModel):
    summary: str

class AdvisorState(TypedDict):
    age: int
    career: str
    monthly_income: float
    goal_description: str
    goal_amount: float
    goal_years: float

    risk_profile: Optional[RiskProfile]
    search_results: Optional[list]
    candidate_funds: Optional[FundList]
    plan: Optional[InvestmentPlan]
    final_recommendation: Optional[FinalRecommendation]

def search_funds(query: str, max_results: int = 5) -> list[dict]:
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"[search_funds] failed: {e}")
        return []

def safe_calculate(expression: str) -> float:
    """Evaluates arithmetic only — no arbitrary code execution, unlike eval()."""
    try:
        return float(numexpr.evaluate(expression))
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")
    
def profiling_agent(state: AdvisorState) -> AdvisorState:
    structured_llm = llm.with_structured_output(RiskProfile)
    prompt = f"""Assess the investment risk profile for this person:
Age: {state['age']}
Career: {state['career']}
Monthly income: {state['monthly_income']}
Goal: {state['goal_description']}, target {state['goal_amount']} in {state['goal_years']} years.

Classify their risk tolerance and explain your reasoning."""

    result = structured_llm.invoke(prompt)
    state["risk_profile"] = result
    return state

def research_agent(state: AdvisorState) -> AdvisorState:
    query = f"best {state['risk_profile'].risk_category} mutual funds India 2026"
    raw_results = search_funds(query)
    state["search_results"] = raw_results

    if not raw_results:
        state["candidate_funds"] = FundList(funds=[])
        return state

    context = "\n\n".join(
        f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r.get('body','')}"
        for r in raw_results
    )

    structured_llm = llm.with_structured_output(FundList)
    prompt = f"""Extract actual fund names from these search results.
Risk profile to match: {state['risk_profile'].risk_category}

{context}

Only include funds that genuinely fit this risk profile. If none clearly qualify, return an empty list."""

    result = structured_llm.invoke(prompt)
    state["candidate_funds"] = result
    return state

def planning_agent(state: AdvisorState) -> AdvisorState:
    structured_llm = llm.with_structured_output(InvestmentPlan)
    fund_names = [f.fund_name for f in state["candidate_funds"].funds]

    prompt = f"""Given these candidate funds: {fund_names}
Risk profile: {state['risk_profile'].risk_category}
Goal: {state['goal_amount']} in {state['goal_years']} years.

Estimate a realistic expected annual return % for this risk profile and fund type,
then calculate the required monthly investment using the future value of an
annuity formula: FV = P * [((1+r)^n - 1) / r], solved for P.
r = monthly rate, n = months. Show your reasoning, including whether the goal
looks realistic given the assumed return."""

    result = structured_llm.invoke(prompt)
    state["plan"] = result
    return state

def aggregator_agent(state: AdvisorState) -> AdvisorState:
    structured_llm = llm.with_structured_output(FinalRecommendation)
    prompt = f"""Summarize this investment plan for the user in plain language:
Risk profile: {state['risk_profile'].risk_category} ({state['risk_profile'].reasoning})
Recommended funds: {[f.fund_name for f in state['candidate_funds'].funds]}
Monthly investment needed: {state['plan'].monthly_investment}
Assumed return: {state['plan'].expected_annual_return_pct}%

Include a brief disclaimer that verify once."""

    result = structured_llm.invoke(prompt)
    state["final_recommendation"] = result
    return state

graph = StateGraph(AdvisorState)
graph.add_node("profiling", profiling_agent)
graph.add_node("research", research_agent)
graph.add_node("planning", planning_agent)
graph.add_node("aggregator", aggregator_agent)

graph.add_edge(START, "profiling")
graph.add_edge("profiling", "research")
graph.add_edge("research", "planning")
graph.add_edge("planning", "aggregator")
graph.add_edge("aggregator", END)

app = graph.compile()

def get_int(prompt: str, min_val: int = None, max_val: int = None) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if min_val is not None and val < min_val:
            print(f"Value must be at least {min_val}.")
            continue
        if max_val is not None and val > max_val:
            print(f"Value must be at most {max_val}.")
            continue
        return val

def get_float(prompt: str, min_val: float = None) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if min_val is not None and val <= min_val:
            print(f"Value must be greater than {min_val}.")
            continue
        return val

def get_str(prompt: str) -> str:
    while True:
        raw = input(prompt).strip()
        if raw:
            return raw
        print("This field can't be empty.")

def collect_user_input() -> AdvisorState:
    print("=== Investment Advisor Setup ===")
    age = get_int("Your age: ", min_val=18, max_val=100)
    career = get_str("Your career/profession: ")
    monthly_income = get_float("Your monthly income: ", min_val=0)
    goal_description = get_str("What are you saving for? (e.g. 'buy a house'): ")
    goal_amount = get_float("Target amount for this goal: ", min_val=0)
    goal_years = get_float("Years to reach this goal: ", min_val=0)

    return {
        "age": age,
        "career": career,
        "monthly_income": monthly_income,
        "goal_description": goal_description,
        "goal_amount": goal_amount,
        "goal_years": goal_years,
        "risk_profile": None,
        "search_results": None,
        "candidate_funds": None,
        "plan": None,
        "final_recommendation": None,
    }

if __name__ == "__main__":
    initial_state = collect_user_input()
    result = app.invoke(initial_state)
    print("\n=== Recommendation ===")
    print(result["final_recommendation"].summary)