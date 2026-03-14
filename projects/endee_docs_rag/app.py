import gradio as gr
from sentence_transformers import SentenceTransformer
from endee import Endee

print("Initializing Endee client and SentenceTransformer model...")
client = Endee()

INDEX_NAME = "endee_docs"

try:
    index = client.get_index(name=INDEX_NAME)
except Exception as e:
    print(f"Error connecting to Endee index '{INDEX_NAME}'. Did you run indexer.py first?")
    index = None

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query):
    if index is None:
        return "Error: Endee Index not found. Please run indexer.py to build the knowledge base."
        
    # Embed the query
    vector = model.encode(query).tolist()
    
    # Search Endee Index
    try:
        results = index.query(vector=vector, top_k=5)
    except Exception as e:
        return f"Error during query: {e}"
    
    # Format the results
    formatted_results = []
    
    for count, res in enumerate(results, start=1):
        # Depending on Endee python SDK return types
        meta = res.get('meta', {}) if isinstance(res, dict) else getattr(res, 'meta', {})
        source = meta.get('source', 'Unknown')
        text = meta.get('text', '')
        sim = res.get('similarity', 0.0) if isinstance(res, dict) else getattr(res, 'similarity', 0.0)
        
        formatted_results.append(
            f"### {count}. Source: `{source}` (Similarity: **{sim:.3f}**)\n\n"
            f"> {text}\n"
        )
        
    if not formatted_results:
        return "No relevant documents found."
        
    return "\n---\n".join(formatted_results)

# Create Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), css=".container { max-width: 800px; margin: auto; }") as demo:
    gr.Markdown(
        """
        # 📚 Endee Semantic Search & RAG Demo
        This project demonstrates how to use the **Endee Vector Database** as the primary storage and retrieval engine for a generic Knowledge Base Application. 
        It embeds markdown documents using `sentence-transformers` and performs sub-millisecond similarity search over them.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Ask a question about Endee", 
                placeholder="e.g. How do I install Endee with Docker?", 
                lines=1
            )
        with gr.Column(scale=1):
            search_button = gr.Button("Search Knowledge Base", variant="primary")
            
    results_output = gr.Markdown(label="Retrieval Results")
    
    search_button.click(fn=search, inputs=query_input, outputs=results_output)
    query_input.submit(fn=search, inputs=query_input, outputs=results_output)
    
    gr.Markdown(
        """
        ---
        **Powered by [Endee.io](https://endee.io) | Agentic AI Workflows**
        """
    )

if __name__ == "__main__":
    print("Starting Gradio Web APP...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
