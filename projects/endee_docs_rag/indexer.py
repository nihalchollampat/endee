import os
import glob
import re
from bs4 import BeautifulSoup
import markdown
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# Initialize the Endee client
client = Endee()

INDEX_NAME = "endee_docs"
DIMENSION = 384  # For all-MiniLM-L6-v2

def setup_index():
    print("Setting up index...")
    try:
        # Create a new index
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",
            precision=Precision.INT8
        )
        print("Created index.")
    except Exception as e:
        print(f"Index might already exist or error occurred: {e}")
        pass
    
    return client.get_index(name=INDEX_NAME)

def clean_markdown(md_text):
    # Convert markdown to HTML
    html = markdown.markdown(md_text)
    # Extract text from HTML
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks

def build_knowledge_base(index, knowledge_dir="knowledge"):
    print(f"Loading files from {knowledge_dir}...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    md_files = glob.glob(os.path.join(knowledge_dir, "*.md"))
    
    docs_to_insert = []
    chunk_id = 0
    
    for filepath in md_files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        clean_text = clean_markdown(content)
        chunks = chunk_text(clean_text, chunk_size=100, overlap=20)
        
        for chunk in chunks:
            # Generate embedding
            embedding = model.encode(chunk).tolist()
            
            docs_to_insert.append({
                "id": f"{filename}_chunk{chunk_id}",
                "vector": embedding,
                "meta": {
                    "source": filename,
                    "text": chunk
                }
            })
            chunk_id += 1
            
    print(f"Inserting {len(docs_to_insert)} chunks into Endee...")
    # Process upsert in batches of 100
    batch_size = 100
    for i in range(0, len(docs_to_insert), batch_size):
        batch = docs_to_insert[i:i+batch_size]
        index.upsert(batch)
        print(f"Inserted batch {i//batch_size + 1}/{(len(docs_to_insert) + batch_size - 1)//batch_size}")
        
    print("Done building knowledge base.")

if __name__ == "__main__":
    index = setup_index()
    build_knowledge_base(index)
