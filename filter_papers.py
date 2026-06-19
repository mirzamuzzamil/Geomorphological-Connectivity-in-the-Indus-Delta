import json

keywords = [
    "remote sensing", "earth observation", "satellite", "hydrology", "hydrological", 
    "water", "flood", "wetland", "transformer", "physics", "streamflow", "runoff",
    "soil moisture", "precipitation", "radar", "sar", "landsat", "sentinel", "dem",
    "explainable", "interpretability", "prithvi", "clay", "earthformer", "foundation model",
    "state space", "state-space", "mamba", "lstm", "river", "delta"
]

def is_relevant(paper):
    title = paper["title"].lower()
    abstract = paper["abstract"].lower() if paper["abstract"] else ""
    journal = paper["journal"].lower()
    
    # Exclude medical, biology (non-environmental), computer science unrelated, social science unrelated
    exclude_keywords = [
        "covid", "cancer", "clinical", "dementia", "alzheimer", "wireless", "telecom", "cellular", 
        "blockchain", "microbiota", "fatty liver", "molecular", "gene", "protein", "cryptography",
        "wireless networks", "edge computing", "traffic", "vehicular", "6g", "uav"
    ]
    
    for ex in exclude_keywords:
        if ex in title or ex in journal:
            return False
            
    # Must contain at least one positive keyword
    for kw in keywords:
        if kw in title or kw in abstract or kw in journal:
            return True
    return False

def main():
    with open("data/papers.json", "r") as f:
        papers = json.load(f)
        
    relevant = [p for p in papers if is_relevant(p)]
    print(f"Total papers: {len(papers)}, Relevant papers: {len(relevant)}")
    
    # Print the top 40 relevant papers
    sorted_relevant = sorted(relevant, key=lambda x: x["citations"], reverse=True)
    for i, p in enumerate(sorted_relevant[:30]):
        print(f"{i+1}. {p['citations']} | {p['year']} | {p['journal']} | {p['title']}")
        
    with open("data/relevant_papers.json", "w") as f:
        json.dump(relevant, f, indent=2)

if __name__ == "__main__":
    main()
