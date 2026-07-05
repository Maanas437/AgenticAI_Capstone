# Investment Advisor — Multi-Agent System

A capstone project for Agentic AI Learners' Space 2026 (Week 4). It takes basic info about a person, age, career, income, and a savings goal, and produces a risk profile, a shortlist of candidate funds, a required monthly investment amount, and a plain-language summary.

## Why four agents

1. **Profiling Agent**: classifies risk tolerance from unstructured inputs.
2. **Research Agent**: searches the web and filters results against the risk profile.
3. **Planning Agent**: estimates a realistic return and calculates required monthly investment.
4. **Aggregator Agent**: writes the final summary with a disclaimer.

They're connected with **LangGraph** as a straight **Pipeline**. Each agent runs once, in fixed order. This is not a Supervisor pattern; there's no branching or conditional re-routing here.

## Stack

- LLM: Gemini 2.5 Flash, via `langchain-google-genai`.
- Search: DuckDuckGo, via the `ddgs` package (the current maintained name; the older `duckduckgo_search` package is frozen).
- Structured output: `with_structured_output()` with Pydantic schemas. Under the hood this still uses JSON Schema; Pydantic just removes manual parsing from the code.
- Math safety: `numexpr` instead of `eval()`, for its restricted grammar, not for speed.

## Known limitations

- DuckDuckGo search isn't officially supported for automated use. It can rate-limit without warning.
- Search results aren't filtered for recency. Old articles can surface for current-year queries.
- Not licensed financial advice. Output includes a disclaimer for this reason.
- No MCP or RAG in this version, though both were part of this week's syllabus. See "Possible extensions."

## Project structure

```
invest-advisor-agent/
├── main.py
├── agents.py
├── tools.py
├── schemas.py
├── state.py
├── .env
└── requirements.txt
```

## Setup

Requires Python 3.11 and a Google API key with Gemini access.

```bash
git clone <your-repo-url>
cd invest-advisor-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_API_KEY=your_key_here
```

Never hardcode the key in source. If a real key is ever committed, treat it as compromised and regenerate it.

## Running it

```bash
python main.py
```

You'll be prompted for age, career, monthly income, savings goal, target amount, and time horizon. The pipeline runs through all four agents and prints a recommendation with the required monthly investment and reasoning.

## Possible extensions

- Wrap `search_funds` and `safe_calculate` behind an MCP server instead of plain LangChain tools.
- Add a RAG layer for fund category knowledge (fees, volatility ranges) instead of relying on the LLM's own knowledge.
- Add a conditional edge so Planning can send the state back to Research if the goal looks unrealistic. This would move the pattern from Pipeline toward Supervisor.

## Disclaimer

This is a student project. It is not a substitute for advice from a licensed financial advisor. Verify any fund data independently before acting on it.
