import json
import re

def categorize_paper(paper):
    title = paper["title"].lower()
    abstract = paper["abstract"].lower() if paper["abstract"] else ""
    journal = paper["journal"].lower()
    
    # Check for exclusion
    exclude_keywords = [
        "covid", "cancer", "clinical", "dementia", "alzheimer", "wireless", "telecom", "cellular", 
        "blockchain", "microbiota", "fatty liver", "molecular", "gene", "protein", "cryptography",
        "wireless networks", "edge computing", "traffic", "vehicular", "6g", "uav", "medical",
        "brain", "heart", "tumor", "syndrome", "therapy", "clinical trials", "nanoparticle",
        "pathogen", "bacterial", "viral", "genome", "genomic", "chromatin", "histone"
    ]
    
    for ex in exclude_keywords:
        if ex in title or ex in journal:
            return None

    # Categories
    # 1. Earth Observation Foundation Models & Remote Sensing Transformers (EO-FM)
    if any(k in title or k in abstract for k in ["foundation model", "prithvi", "clay model", "earthformer", "transformer", "attention-based", "vit", "vision transformer"]):
        if any(k in title or k in abstract or k in journal for k in ["remote sensing", "earth observation", "satellite", "geospatial", "climate", "weather"]):
            return "EO-FM"
            
    # 2. Physics-Informed Learning (PINN)
    if any(k in title or k in abstract for k in ["physics-informed", "physics informed", "pinn", "physical constraint", "mass balance", "water balance", "conservation law"]):
        return "PINN"
        
    # 3. Hydrological AI & Streamflow/Runoff Modeling (Hydro-AI)
    if any(k in title or k in abstract for k in ["hydrology", "hydrological", "streamflow", "runoff", "discharge", "rainfall-runoff", "lstm", "soil moisture", "smap", "era5-land", "groundwater"]):
        return "Hydro-AI"
        
    # 4. Water Occurrence & Flood Mapping (Flood-Water)
    if any(k in title or k in abstract for k in ["flood mapping", "flood detection", "surface water", "water occurrence", "jrc", "sentinel-1", "s1", "s-1", "backscatter", "ndwi", "mndwi"]):
        return "Flood-Water"
        
    # 5. Wetland Mapping & Monitoring (Wetland)
    if any(k in title or k in abstract for k in ["wetland", "mangrove", "peatland", "swamp", "marsh", "ramsar"]):
        return "Wetland"
        
    # 6. Explainable AI (XAI) in Hydrology/Geoscience
    if any(k in title or k in abstract for k in ["explainable", "interpretability", "shap", "shapley", "lime", "integrated gradients", "attribution", "feature importance"]):
        return "XAI"
        
    return None

def main():
    with open("data/papers.json", "r") as f:
        papers = json.load(f)
        
    categorized = {
        "EO-FM": [],
        "PINN": [],
        "Hydro-AI": [],
        "Flood-Water": [],
        "Wetland": [],
        "XAI": []
    }
    
    for p in papers:
        cat = categorize_paper(p)
        if cat:
            categorized[cat].append(p)
            
    total = 0
    for cat, plist in categorized.items():
        # Sort each list by citation count descending
        categorized[cat] = sorted(plist, key=lambda x: x["citations"], reverse=True)
        print(f"Category {cat}: {len(categorized[cat])} papers")
        total += len(categorized[cat])
        
    print(f"Total categorized papers: {total}")
    
    # We want to select at least 50 papers total, balancing across categories
    selected_papers = []
    # Let's take the top papers from each category to build a solid 50+ list
    for cat, plist in categorized.items():
        # take up to 10-12 papers from each category
        selected_papers.extend(plist[:10])
        
    # If we need more to reach 55-60, take the next highest cited ones from the pool
    if len(selected_papers) < 55:
        already_selected = {p["id"] for p in selected_papers}
        pool = []
        for cat, plist in categorized.items():
            for p in plist:
                if p["id"] not in already_selected:
                    pool.append(p)
        pool = sorted(pool, key=lambda x: x["citations"], reverse=True)
        selected_papers.extend(pool[:(55 - len(selected_papers))])
        
    print(f"Selected {len(selected_papers)} papers for literature matrix.")
    
    # Format and save
    with open("data/selected_papers.json", "w") as f:
        json.dump(selected_papers, f, indent=2)
        
    # Write a quick markdown summary
    with open("data/literature_summary.md", "w") as f:
        f.write("# Selected Literature Summary\n\n")
        for i, p in enumerate(selected_papers):
            f.write(f"{i+1}. **{p['title']}** ({p['year']}) - {p['journal']}\n")
            f.write(f"   * Authors: {', '.join(p['authors'][:3]) if p['authors'] else 'N/A'}\n")
            f.write(f"   * Citations: {p['citations']} | DOI: {p['doi']}\n\n")

if __name__ == "__main__":
    main()
