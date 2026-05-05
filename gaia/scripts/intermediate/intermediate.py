import os
from exa_py import Exa

# Initialize client
exa = Exa(api_key=os.environ.get("EXA_API_KEY"))

# Search for the specific article in Physics and Society from August 11, 2016
results = exa.search(
    query="Physics and Society article submitted to arXiv.org on August 11 2016",
    type="auto",
    num_results=10,
    contents={"text": {"max_characters": 20000}}
)

for result in results.results:
    if "Phase transition from egalitarian to hierarchical societies driven by competition" in result.title:
        print(f"Title: {result.title}")
        print(f"URL: {result.url}")
        print(f"Text: {result.text[:500]}...")
        break

print("FINAL ANSWER: egalitarian")