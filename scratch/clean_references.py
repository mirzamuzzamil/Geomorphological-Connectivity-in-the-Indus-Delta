import os
import re

def get_cited_keys():
    base_dir = "prism"
    tex_files = []
    
    # We can scan all tex files in prism/ sections, main_split.tex, and supplementary.tex
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))
                
    cited_keys = set()
    for tf in tex_files:
        with open(tf, "r") as f:
            content = f.read()
        # Find citations: \cite{key}, \citep{key}, \citet{key}, \citep[...][...]{key}
        # Matches multiple comma-separated keys like \cite{key1,key2}
        cite_matches = re.findall(r'\\cite[a-z]*\s*(?:\[[^\]]*\])*\s*\{\s*([^}]+)\s*\}', content)
        for match in cite_matches:
            for key in match.split(','):
                cited_keys.add(key.strip())
                
    return cited_keys

def clean_bib():
    cited_keys = get_cited_keys()
    bib_path = "prism/references.bib"
    
    if not os.path.exists(bib_path):
        print("references.bib not found!")
        return
        
    with open(bib_path, "r") as f:
        content = f.read()
        
    # Split bib file into entries
    # Simple regex to split by entries: @type{key, ...}
    entries = re.split(r'\n(?=@)', content)
    
    cleaned_entries = []
    removed_keys = []
    
    header = ""
    # Process the first part if it's comment/header
    if len(entries) > 0 and not entries[0].strip().startswith("@"):
        header = entries[0]
        entries = entries[1:]
        
    for entry in entries:
        match = re.match(r'@\w+\s*\{\s*([\w\-]+)\s*,', entry)
        if match:
            key = match.group(1).strip()
            if key in cited_keys:
                cleaned_entries.append(entry)
            else:
                removed_keys.append(key)
        else:
            cleaned_entries.append(entry)
            
    # Write back
    new_content = header + "\n".join(cleaned_entries)
    with open(bib_path, "w") as f:
        f.write(new_content)
        
    print(f"Total cited keys: {len(cited_keys)}")
    print(f"Removed {len(removed_keys)} unused keys: {removed_keys}")

if __name__ == "__main__":
    clean_bib()
