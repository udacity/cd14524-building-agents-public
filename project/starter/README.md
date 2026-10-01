# UdaPlay - AI Game Research Agent Project

## Project Overview
UdaPlay is an AI-powered research agent for the video game industry. This project is divided into two main parts that will help you build a sophisticated AI agent capable of answering questions about video games using both local knowledge and web searches.

## Project Structure

### Part 1: Offline RAG (Retrieval-Augmented Generation)
In this part, you'll build a Vector Database using ChromaDB to store and retrieve video game information efficiently.

Key tasks:
- Set up ChromaDB as a persistent client
- Create a collection with appropriate embedding functions
- Process and index game data from JSON files
- Each game document contains:
  - Name
  - Platform
  - Genre
  - Publisher
  - Description
  - Year of Release

### Part 2: AI Agent Development
Build an intelligent agent that combines local knowledge with web search capabilities.

The agent will have the following capabilities:
1. Answer questions using internal knowledge (RAG)
2. Search the web when needed
3. Maintain conversation state
4. Return structured outputs
5. Store useful information for future use

Required Tools to Implement:
1. `retrieve_game`: Search the vector database for game information
2. `evaluate_retrieval`: Assess the quality of retrieved results
3. `game_web_search`: Perform web searches for additional information

## Requirements

### Environment Setup
Create a `.env` file in `project/starter` (copy `.env.example` from the repo root) with:
```
OPENAI_API_KEY="voc-..."          # the voc- key from the classroom's Cloud Resources panel
OPENAI_BASE_URL="https://openai.vocareum.com/v1"
TAVILY_API_KEY="tvly-..."
```
The file must be named exactly `.env`.

### Project Dependencies
- Python 3.11+ (the Udacity workspace runs 3.13)
- `pip install -r requirements.txt` installs ChromaDB, OpenAI, Pydantic, python-dotenv, Tavily and pdfplumber
- The `pysqlite3` cell at the top of each notebook is only for the Udacity workspace; locally it does nothing

### Directory Structure
```
project/
├── starter/
│   ├── games/           # JSON files with game data
│   ├── lib/             # Custom library implementations
│   │   ├── llm.py       # LLM abstractions
│   │   ├── messages.py  # Message handling
│   │   ├── ...
│   │   └── tooling.py   # Tool implementations
│   ├── Udaplay_01_starter_project.ipynb  # Part 1 implementation
│   └── Udaplay_02_starter_project.ipynb  # Part 2 implementation
```

## Getting Started

1. Create and activate a virtual environment
2. `pip install -r requirements.txt`
3. Set up your `.env` file in `project/starter`
4. Run both notebooks from `project/starter`, in order:
   - Part 1 builds the vector database in `chromadb/` with a collection named `udaplay`
   - Part 2 loads that same path and collection, so keep both names and the embedding function unchanged, and don't delete `chromadb/`

## Testing Your Implementation

After completing both parts, test your agent with questions like:
- "When was Pokémon Gold and Silver released?"
- "Which one was the first 3D platformer Mario game?"
- "Was Mortal Kombat X released for PlayStation 5?"

## Advanced Features (Stand Out, optional)

After completing the basic implementation, you can enhance your agent with:
- Long-term memory that persists across sessions (note: `lib/memory.py`'s `LongTermMemory` is in-memory unless you give it a persistent ChromaDB client)
- Additional tools and capabilities

## Notes
- Make sure to implement proper error handling
- Follow best practices for API key management
- Document your code thoroughly
- Test your implementation with various types of queries
