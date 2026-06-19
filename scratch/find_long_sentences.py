import os
import re

def find_long_sentences(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.tex')]
    for file in files:
        path = os.path.join(directory, file)
        with open(path, 'r') as f:
            content = f.read()
        
        # Remove comments
        content = re.sub(r'%.*?\n', '\n', content)
        
        # Clean text a bit (remove LaTeX commands like \citep{...})
        content_cleaned = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', content)
        content_cleaned = re.sub(r'\\[a-zA-Z]+', '', content_cleaned)
        
        # Split into sentences using simple regex
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', content_cleaned)
        
        print(f"\nFile: {file}")
        long_count = 0
        for sent in sentences:
            sent_trimmed = sent.strip().replace('\n', ' ')
            words = [w for w in sent_trimmed.split(' ') if w]
            if len(words) > 40:
                print(f"  [{len(words)} words]: {sent_trimmed[:100]}...")
                long_count += 1
        print(f"  Found {long_count} sentences exceeding 40 words.")

if __name__ == "__main__":
    find_long_sentences("prism/sections")
