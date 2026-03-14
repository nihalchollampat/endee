import argparse
import time
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer

# Toy dataset of movies
movies = [
    {"id": "mv1", "title": "Inception", "plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.", "year": 2010, "genre": "Sci-Fi"},
    {"id": "mv2", "title": "The Matrix", "plot": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.", "year": 1999, "genre": "Sci-Fi"},
    {"id": "mv3", "title": "Interstellar", "plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.", "year": 2014, "genre": "Sci-Fi"},
    {"id": "mv4", "title": "The Godfather", "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.", "year": 1972, "genre": "Crime"},
    {"id": "mv5", "title": "The Dark Knight", "plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.", "year": 2008, "genre": "Action"},
    {"id": "mv6", "title": "Pulp Fiction", "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.", "year": 1994, "genre": "Crime"},
    {"id": "mv7", "title": "Forrest Gump", "plot": "The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate and other historical events unfold through the perspective of an Alabama man with an IQ of 75, whose only desire is to be reunited with his childhood sweetheart.", "year": 1994, "genre": "Drama"},
    {"id": "mv8", "title": "Gladiator", "plot": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.", "year": 2000, "genre": "Action"},
    {"id": "mv9", "title": "Titanic", "plot": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.", "year": 1997, "genre": "Romance"},
    {"id": "mv10", "title": "Avatar", "plot": "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.", "year": 2009, "genre": "Sci-Fi"},
    {"id": "mv11", "title": "The Shawshank Redemption", "plot": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.", "year": 1994, "genre": "Drama"},
    {"id": "mv12", "title": "Fight Club", "plot": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into much more.", "year": 1999, "genre": "Drama"},
    {"id": "mv13", "title": "Goodfellas", "plot": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito in the Italian-American crime syndicate.", "year": 1990, "genre": "Crime"}
]

def load_data(client, model, index_name="movie_recommendations"):
    # Delete index if exists, to ensure fresh start (optional based on Endee API)
    try:
        # Some SDKs implement delete_index, we'll just try
        client.delete_index(name=index_name)
        print(f"Dropped existing index '{index_name}'")
    except Exception:
        pass

    # Create Index
    print(f"Creating Endee index '{index_name}' with 384 dimensions...")
    client.create_index(
        name=index_name,
        dimension=384,
        space_type="cosine",
        precision=Precision.FP32   # using FP32 (or INT8) for vector encoding
    )
    
    index = client.get_index(name=index_name)
    
    # Process & upsert vectors
    print("Generating embeddings for movies and upserting into Endee...")
    upsert_data = []
    for m in movies:
        vec = model.encode(m['plot']).tolist()
        upsert_data.append({
            "id": m["id"],
            "vector": vec,
            "meta": {
                "title": m["title"],
                "plot": m["plot"],
                "year": m["year"],
                "genre": m["genre"]
            }
        })
        
    index.upsert(upsert_data)
    print("Successfully populated the vector database with movie plots!\n")
    return index

def query_loop(index, model):
    print("=========================================================")
    print("           Semantic Movie Recommender Started            ")
    print("=========================================================")
    print("Enter a description of what kind of movie you want to see.")
    print("Example: 'A space adventure with exploration'")
    print("---------------------------------------------------------")
    
    while True:
        try:
            query = input("\nYour movie preference (or type 'quit' to exit): ")
            if query.lower().strip() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            if not query.strip():
                continue
                
            start_time = time.time()
            query_vector = model.encode(query).tolist()
            results = index.query(vector=query_vector, top_k=3)
            search_time = time.time() - start_time
            
            print(f"\nTop 3 results in {search_time:.4f}s:")
            if not results:
                print("No matches found.")
                continue
                
            for i, r in enumerate(results):
                # Safely parsing response objects or dicts based on Endee python SDK version
                if isinstance(r, dict):
                    meta = r.get("meta", r.get("metadata", {}))
                    sim = r.get("similarity", r.get("score", 0.0))
                else:
                    meta = getattr(r, "meta", getattr(r, "metadata", {}))
                    sim = getattr(r, "similarity", getattr(r, "score", 0.0))
                
                print(f"{i+1}. {meta.get('title')} ({meta.get('year')}) - {meta.get('genre')}")
                print(f"   Similarity: {sim:.4f}")
                print(f"   Plot: {meta.get('plot')}")
                print("   ---")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred during query: {e}")

def main():
    parser = argparse.ArgumentParser(description="Endee Semantic Movie Search Recommender")
    parser.add_argument("--endee-url", type=str, default="http://localhost:8080/api/v1", help="Endee Server API URL")
    args = parser.parse_args()

    print("Loading Sentence Transformer model ('all-MiniLM-L6-v2', dims=384)...")
    # This model runs fast locally and produces good semantic representations
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Connecting to Endee server at {args.endee_url}...")
    try:
        client = Endee()
        # client.set_base_url(args.endee_url) # Set Base URL if needed by Endee sdk
    except Exception as e:
        print(f"Failed to initialize Endee Client: {e}")
        return

    index_name = "movie_recommendations"
    try:
        index = client.get_index(name=index_name)
        print("Existing index found. We'll drop & recreate for the demo.")
        load_data(client, model, index_name)
    except Exception:
        # Index doesn't exist, create it
        load_data(client, model, index_name)
        
    index = client.get_index(name=index_name)
    query_loop(index, model)

if __name__ == "__main__":
    main()
