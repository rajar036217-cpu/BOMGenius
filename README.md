# BOMGenius

### Developer guide
- Setup uv
- Install all dependencies including dev run: 
`uv sync
`
- In vs code install black python formatter

### To run
```bash 
cd BOMGenius/BOMGenius2.0/backend
uv run uvicorn main:app --reload --port 8000
```

### TODO
Action items:
- Start using logger
- port appication to docker for easy run
- update requirements.txt for better requirements handling
- update Readme.md with the commands need to be run.
- Specify source directory from where i can run the application
- Code cleanups :(
- Need to know the pre requirements like ollama model (better to have setup.py)
- All configurables should be in one place use "loadenv" module
- Unit and regression tests required

Front-end
- Better to use single page application (not required right away but will be helpful for learning)