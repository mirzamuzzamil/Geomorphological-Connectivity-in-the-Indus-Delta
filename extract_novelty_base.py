import json

def main():
    with open("data/papers.json", "r") as f:
        papers = json.load(f)
        
    print(f"Total papers: {len(papers)}")
    
    # We want to select the top 30 papers that are most relevant to:
    # 1. Foundation Models / Earth Observation Transformers (Prithvi, Clay, Earthformer, etc.)
    # 2. Physics-Informed Neural Networks (PINN) in hydrology
    # 3. Deep Learning Streamflow/Runoff (LSTM, etc.)
    # 4. Explainability (SHAP, LIME, attention in geosciences)
    # 5. Flood / Water Mapping (JRC, Sentinel-1/2, Pekel, GEE)
    
    # Let's filter and score each paper's relevance to EHSFM.
    relevant_scored = []
    
    keywords = {
        "fm_transformer": ["foundation model", "transformer", "attention", "prithvi", "clay", "earthformer", "vit", "mamba", "state space", "state-space"],
        "pinn": ["physics", "physical", "mass balance", "mass-balance", "conservation", "differentiable", "constraint"],
        "hydrology": ["hydrology", "hydrological", "streamflow", "runoff", "discharge", "catchment", "basin", "smap", "soil moisture", "lstm"],
        "water_flood": ["flood", "surface water", "occurrence", "inundation", "jrc", "sentinel-1", "sentinel-2", "landsat", "wetland", "mangrove", "swamp"]
    }
    
    for p in papers:
        title = p["title"].lower()
        abstract = p["abstract"].lower() if p["abstract"] else ""
        journal = p["journal"].lower()
        
        # Check exclusions
        exclude_keywords = [
            "covid", "cancer", "clinical", "dementia", "alzheimer", "wireless", "telecom", "cellular", 
            "blockchain", "microbiota", "fatty liver", "molecular", "gene", "protein", "cryptography",
            "networks", "6g", "uav", "medical", "brain", "heart", "tumor", "syndrome", "therapy",
            "pathogen", "bacterial", "viral", "genome", "genomic", "chromatin", "histone"
        ]
        if any(ex in title or ex in journal for ex in exclude_keywords):
            continue
            
        score = 0
        matches = []
        for cat, kw_list in keywords.items():
            cat_score = 0
            for kw in kw_list:
                if kw in title:
                    cat_score += 10
                if kw in abstract:
                    cat_score += 2
                if kw in journal:
                    cat_score += 5
            if cat_score > 0:
                score += cat_score
                matches.append(cat)
                
        # Boost papers with multiple categories (e.g. hydrology + physics or hydrology + transformer)
        if len(matches) > 1:
            score += 15 * len(matches)
            
        if score > 5:
            relevant_scored.append((score, p))
            
    relevant_scored.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Scored {len(relevant_scored)} papers.")
    for i, (score, p) in enumerate(relevant_scored[:40]):
        print(f"{i+1}. Score={score} | Citations={p['citations']} | {p['year']} | {p['journal']} | {p['title']}")
        
    # Save the top 40 papers
    top_papers = [x[1] for x in relevant_scored[:40]]
    with open("data/top_40_relevant.json", "w") as f:
        json.dump(top_papers, f, indent=2)

if __name__ == "__main__":
    main()
