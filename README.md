# Travel Planner

An intelligent travel planning application that uses the AG2 framework to coordinate multiple specialized AI agents for creating optimized travel itineraries within budget constraints. Mock data as well as a locally run LLM were used instead of integrating with APIs so as to simplify the application. Please refer to the AG2 repository for more details: [AG2: Open-Source AgentOS for AI Agents](https://github.com/ag2ai/ag2)

## Architecture

The system uses three specialized AG2 agents:

1. **Flight Agent**: Analyzes flight options and recommends based on price, duration, and convenience
2. **Hotel Agent**: Evaluates accommodations considering location, rating, and total stay cost
3. **Activity Agent**: Creates daily itineraries mixing free and paid activities

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Mistral model downloaded in Ollama (`ollama pull mistral`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/zacharyzusin/TravelPlanner.git
```

2. Install dependencies:
```bash
pip install "ag2[openai]"
```

3. Ensure Ollama is running:
```bash
ollama serve
```

4. Run the example trip planner:

```bash
python travel.py
```
