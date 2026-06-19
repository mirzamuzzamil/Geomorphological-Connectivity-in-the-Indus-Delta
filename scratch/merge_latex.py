import os
import re

def expand_inputs(file_path, base_dir):
    if not os.path.exists(file_path):
        # Try appending .tex extension
        if not file_path.endswith(".tex"):
            file_path += ".tex"
            
    if not os.path.exists(file_path):
        # Try finding relative to base_dir
        alt_path = os.path.join(base_dir, file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        elif os.path.exists(alt_path + ".tex"):
            file_path = alt_path + ".tex"
        else:
            raise FileNotFoundError(f"Could not find input file: {file_path}")
        
    with open(file_path, "r") as f:
        content = f.read()
        
    def replace_match(match):
        rel_path = match.group(1).strip()
        rel_path = rel_path.strip('"').strip("'")
        if not rel_path.endswith(".tex"):
            rel_path += ".tex"
        
        # Try locating relative to base_dir or file's parent dir
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            # Try directly
            if os.path.exists(rel_path):
                full_path = rel_path
            else:
                # Try relative to the directory of the current file being parsed
                file_dir = os.path.dirname(file_path)
                full_path = os.path.join(file_dir, rel_path)
                
        print(f"Expanding: {rel_path} -> {full_path}")
        return expand_inputs(full_path, base_dir)
        
    # Replace \input{path} recursively
    content = re.sub(r'\\input\s*\{\s*([^}]+)\s*\}', replace_match, content)
    return content

def main():
    base_dir = "prism"
    main_tex = os.path.join(base_dir, "main.tex")
    backup_tex = os.path.join(base_dir, "main_split.tex")
    
    if not os.path.exists(main_tex):
        print(f"Error: main.tex not found at {main_tex}")
        return
        
    if not os.path.exists(backup_tex):
        # Backup original split main.tex
        with open(main_tex, "r") as src:
            split_content = src.read()
        with open(backup_tex, "w") as dst:
            dst.write(split_content)
        print(f"Backed up split main.tex to {backup_tex}")
    else:
        print(f"Using existing split main.tex backup from {backup_tex}")
        
    merged_content = expand_inputs(backup_tex, base_dir)
    
    with open(main_tex, "w") as f:
        f.write(merged_content)
    print(f"Successfully merged all section and table inputs into {main_tex}")

if __name__ == "__main__":
    main()
