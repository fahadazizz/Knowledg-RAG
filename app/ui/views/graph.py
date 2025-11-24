import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from app.tools.graph_store import get_graph_visualization_data
from app.utils.kg_builder import KnowledgeGraphBuilder

def render_graph_page():
    # Clear Graph Button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Clear Graph", type="secondary", use_container_width=True):
            with st.spinner("Clearing knowledge graph..."):
                kg_builder = KnowledgeGraphBuilder()
                kg_builder.clear_graph()
                st.success("Graph cleared successfully!")
                st.rerun()

    # Fetch graph data
    with st.spinner("Fetching graph data..."):
        nodes_data, edges_data = get_graph_visualization_data(limit=100)
    
    if not nodes_data:
        st.warning("No graph data found. Please upload and process documents first.")
    else:
        # Define color palette for different entity types
        color_map = {
            "PERSON": "#FF6B6B",      # Red
            "ORGANIZATION": "#4ECDC4", # Teal
            "LOCATION": "#45B7D1",    # Blue
            "CONCEPT": "#96CEB4",     # Green
            "EVENT": "#FFD93D",       # Yellow
            "TECHNOLOGY": "#6C5CE7",  # Purple
            "PRODUCT": "#A8E6CF",     # Mint
            "Unknown": "#95A5A6"      # Grey
        }
        
        # Create Agraph nodes
        nodes = []
        for n in nodes_data:
            # Get color based on type, default to a fallback if type not found
            node_type = n.get('type', 'Unknown')
            # Handle case where type might be a list or string
            if isinstance(node_type, list):
                node_type = node_type[0] if node_type else 'Unknown'
                
            node_color = color_map.get(node_type, "#667eea") # Default purple-ish
            
            nodes.append(Node(
                id=n['id'], 
                label=n['label'], 
                size=25, 
                color=node_color,
                title=f"Type: {node_type}" # Tooltip
            ))
            
        # Create Agraph edges
        edges = []
        for e in edges_data:
            edges.append(Edge(
                source=e['source'], 
                target=e['target'], 
                label=e['label']
            ))
        
        config = Config(
            width=None, # Auto width
            height=800, # Increased height
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=True
        )
        
        agraph(nodes=nodes, edges=edges, config=config)
