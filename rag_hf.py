# rag_interface.py
import streamlit as st
import pickle
import faiss
import rdflib
import torch
import datetime
import os
import requests
from rdflib import Graph as RDFGraph, Namespace
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# === CONFIGURATION ===
load_dotenv()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EX = Namespace("http://example.org/lang/")

st.set_page_config(
    page_title="Vanishing Voices: Language Atlas",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #e8f4fc;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3498db;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
    .sidebar-title {
        color: #2c3e50;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
    }
    .method-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border-left: 3px solid #3498db;
    }
    .method-title {
        font-weight: 600;
        color: #3498db;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading models and indexes...")
def load_all_components():
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    methods = {}
    for label, suffix, ttl in [
        ("Standard", "", "grafo_ttl_no_hibrido.ttl"),
        ("Hybrid", "_hybrid", "grafo_ttl_hibrido.ttl"),
        ("GraphSAGE", "_hybrid_graphsage", "grafo_ttl_hibrido_graphsage.ttl")
    ]:
        with open(f"id_map{suffix}.pkl", "rb") as f:
            id_map = pickle.load(f)
        with open(f"grafo_embed{suffix}.pickle", "rb") as f:
            G = pickle.load(f)
        index = faiss.read_index(f"index_faiss{suffix}.faiss")
        rdf = RDFGraph()
        rdf.parse(ttl, format="ttl")
        methods[label] = (index, id_map, G, rdf)
    return methods, embedder

methods, embedder = load_all_components()

# === CORE FUNCTIONS ===
def get_top_k(index, id_map, query, k):
    vec = embedder.encode(f"query: {query}", convert_to_tensor=True, device=DEVICE)
    vec = vec.cpu().numpy().astype("float32").reshape(1, -1)
    _, indices = index.search(vec, k)
    return [id_map[i] for i in indices[0]]

def get_context(G, lang_id):
    node = G.nodes.get(lang_id, {})
    lines = [f"**Language:** {node.get('label', lang_id)}"]
    if node.get("wikipedia_summary"):
        lines.append(f"**Wikipedia:** {node['wikipedia_summary']}")
    if node.get("wikidata_description"):
        lines.append(f"**Wikidata:** {node['wikidata_description']}")
    if node.get("wikidata_countries"):
        lines.append(f"**Countries:** {node['wikidata_countries']}")
    return "\n\n".join(lines)

def query_rdf(rdf, lang_id):
    q = f"""
    PREFIX ex: <http://example.org/lang/>
    SELECT ?property ?value WHERE {{ ex:{lang_id} ?property ?value }}
    """
    try:
        return [
            (str(row[0]).split("/")[-1], str(row[1]))
            for row in rdf.query(q)
        ]
    except Exception as e:
        return [("error", str(e))]

def generate_response(index, id_map, G, rdf, user_question, k=3):
    ids = get_top_k(index, id_map, user_question, k)
    context = [get_context(G, i) for i in ids]
    rdf_facts = []
    for i in ids:
        rdf_facts.extend([f"{p}: {v}" for p, v in query_rdf(rdf, i)])
    prompt = f"""<s>[INST]
You are an expert in South American indigenous languages.
Use strictly and only the information below to answer the user question in **English**.
- Do not infer or assume facts that are not explicitly stated.
- If the answer is unknown or insufficient, say "I cannot answer with the available data."
- Limit your answer to 100 words.


### CONTEXT:
{chr(10).join(context)}

### RDF RELATIONS:
{chr(10).join(rdf_facts)}

### QUESTION:
{user_question}

Answer:
[/INST]"""
    try:
        res = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_ID}",
            headers={"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}", "Content-Type": "application/json"},
            json={"inputs": prompt}, timeout=30
        )
        out = res.json()
        if isinstance(out, list) and "generated_text" in out[0]:
            return out[0]["generated_text"].replace(prompt.strip(), "").strip(), ids, context, rdf_facts
        return str(out), ids, context, rdf_facts
    except Exception as e:
        return str(e), ids, context, rdf_facts

# === MAIN FUNCTION ===
def main():
    st.markdown("""
    <h1 class='header'>Vanishing Voices: South America's Endangered Language Atlas</h1>
    <div class='info-box'>
    <b>Linguistic Emergency:</b> Over 40% of South America's indigenous languages face extinction.
    This tool documents these cultural treasures before they disappear forever.
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://glottolog.org/static/img/glottolog_lod.png", width=180)
        
        # About Section
        with st.container():
            st.markdown('<div class="sidebar-title">About This Tool</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="method-card">
                <div class="method-title">Standard Search</div>
                Semantic retrieval based on text-only embeddings. Identifies languages using purely linguistic similarity from Wikipedia summaries and labels.
            </div>
            <div class="method-card">
                <div class="method-title">Hybrid Search</div>
                Combines semantic embeddings with structured data from knowledge graphs. Enriches language representation with contextual facts.
            </div>
            <div class="method-card">
                <div class="method-title">GraphSAGE Search</div>
                Leverages deep graph neural networks to learn relational patterns across languages. Captures complex cultural and genealogical connections.
            </div>
            """, unsafe_allow_html=True)
        
        # Research Settings
        with st.container():
            st.markdown('<div class="sidebar-title">Research Settings</div>', unsafe_allow_html=True)
            k = st.slider("Languages to analyze per query", 1, 10, 3)
            
            st.markdown("**Display Options:**")
            show_ids = st.checkbox("Language IDs", value=True, key="show_ids")
            show_ctx = st.checkbox("Cultural Context", value=True, key="show_ctx")
            show_rdf = st.checkbox("RDF Relations", value=True, key="show_rdf")
        
        # Data Source
        with st.container():
            st.markdown('<div class="sidebar-title">Data Sources</div>', unsafe_allow_html=True)
            st.markdown("""
            - Glottolog
            - Wikidata
            - Wikipedia
            - Ethnologue
            """)

    query = st.text_input("Ask about indigenous languages:", "Which Amazonian languages are most at risk?")

    if st.button("Analyze with All Methods") and query:
        col1, col2, col3 = st.columns(3)
        results = {}
        for col, (label, method) in zip([col1, col2, col3], methods.items()):
            with col:
                st.subheader(f"{label} Analysis")
                start = datetime.datetime.now()
                response, lang_ids, context, rdf_data = generate_response(*method, query, k)
                duration = (datetime.datetime.now() - start).total_seconds()
                st.markdown(response)
                st.markdown(f"⏱️ {duration:.2f}s | 🌐 {len(lang_ids)} languages")
                if show_ids:
                    st.markdown("**Language Identifiers:**")
                    st.code("\n".join(lang_ids))
                if show_ctx:
                    st.markdown("**Cultural Context:**")
                    st.markdown("\n\n---\n\n".join(context))
                if show_rdf:
                    st.markdown("**RDF Knowledge:**")
                    st.code("\n".join(rdf_data))
                results[label] = response

        log = f"""
[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]
QUERY: {query}
STANDARD:
{results.get('Standard', '')}

HYBRID:
{results.get('Hybrid', '')}

GRAPH-SAGE:
{results.get('GraphSAGE', '')}
{'='*60}
"""
        try:
            with open("language_analysis_logs.txt", "a", encoding="utf-8") as f:
                f.write(log)
        except Exception as e:
            st.warning(f"Failed to log: {str(e)}")

if __name__ == "__main__":
    main()