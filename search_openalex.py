import urllib.request
import urllib.parse
import json
import os

def reconstruct_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return ""
    # The abstract_inverted_index is a dict where keys are words and values are lists of positions
    # We want to build the string by placing words at their positions
    word_positions = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join([word for pos, word in word_positions])

def search_openalex(query, filters=None, limit=20):
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": limit,
        "sort": "cited_by_count:desc",
        "mailto": "researcher@example.com"
    }
    
    filter_parts = ["publication_year:2020-2026"]
    if filters:
        filter_parts.append(filters)
    params["filter"] = ",".join(filter_parts)
    
    url = base_url + "?" + urllib.parse.urlencode(params)
    print(f"Querying: {url}")
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get("results", [])
    except Exception as e:
        print(f"Error querying {query}: {e}")
        return []

def main():
    queries = [
        # Foundation Models and EO Transformers
        ("foundation model remote sensing", None),
        ("earth observation transformer", None),
        
        # Hydrological AI & Physics-informed learning
        ("physics informed machine learning hydrology", None),
        ("physics-informed neural network streamflow", None),
        
        # State-space models & latent state learning
        ("state space model hydrology runoff", None),
        ("latent state streamflow LSTM", None),
        
        # Flood mapping
        ("flood mapping Sentinel-1 Sentinel-2 deep learning", None),
        ("surface water occurrence deep learning satellite", None),
        
        # Wetlands and Explainable AI
        ("explainable AI hydrology remote sensing", None),
        ("wetland mapping machine learning remote sensing", None)
    ]
    
    all_papers = {}
    
    for query, filters in queries:
        results = search_openalex(query, filters, limit=12)
        for r in results:
            doi = r.get("doi")
            title = r.get("display_name")
            if not title:
                continue
            
            # Skip if already added
            paper_id = r.get("id")
            if paper_id in all_papers:
                continue
                
            journal = "Unknown Journal"
            loc = r.get("primary_location")
            if loc and loc.get("source"):
                journal = loc["source"].get("display_name", "Unknown Journal")
                
            abstract = reconstruct_abstract(r.get("abstract_inverted_index"))
            
            all_papers[paper_id] = {
                "id": paper_id,
                "title": title,
                "doi": doi,
                "journal": journal,
                "year": r.get("publication_year"),
                "citations": r.get("cited_by_count"),
                "abstract": abstract,
                "authors": [a.get("author", {}).get("display_name") for a in r.get("authorships", [])]
            }
            
    print(f"Retrieved {len(all_papers)} unique papers.")
    
    os.makedirs("data", exist_ok=True)
    with open("data/papers.json", "w") as f:
        json.dump(list(all_papers.values()), f, indent=2)
    print("Saved to data/papers.json")

if __name__ == "__main__":
    main()
