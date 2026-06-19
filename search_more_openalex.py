import urllib.request
import urllib.parse
import json
import os

def reconstruct_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return ""
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
        ("Prithvi NASA foundation model", None),
        ("Clay foundation model remote sensing", None),
        ("Earthformer satellite forecasting", None),
        ("Kratzert LSTM hydrology", None),
        ("physics informed streamflow Chaopeng Shen", None),
        ("explainable AI hydrology SHAP", None),
        ("Mamba state space model hydrology", None),
        ("flood mapping Sentinel-1 deep learning", None),
        ("wetland mapping Sentinel-2 GEE", None),
        ("deep learning surface water occurrence JRC", None),
        ("transformer remote sensing water index", None),
        ("deep learning runoff simulation", None)
    ]
    
    # Load existing papers
    if os.path.exists("data/papers.json"):
        with open("data/papers.json", "r") as f:
            existing = json.load(f)
            all_papers = {p["id"]: p for p in existing}
    else:
        all_papers = {}
        
    for query, filters in queries:
        results = search_openalex(query, filters, limit=15)
        for r in results:
            doi = r.get("doi")
            title = r.get("display_name")
            if not title:
                continue
            
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
            
    print(f"Total unique papers in database: {len(all_papers)}")
    
    with open("data/papers.json", "w") as f:
        json.dump(list(all_papers.values()), f, indent=2)
    print("Saved updated papers to data/papers.json")

if __name__ == "__main__":
    main()
