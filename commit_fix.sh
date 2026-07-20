chmod -R u=rwx,g=rwx,o=rx src/ofa_main.py
git add src/ofa_main.py 
git commit -m "fix(agent): Add auto-feed loops and truncation caps to LLM shell output parser"
git push
