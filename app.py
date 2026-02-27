import streamlit as st
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac
import plotly.express as px
import plotly.graph_objects as go
import time

# --- 1. UI CONFIG & COMMAND CENTER CSS ---
st.set_page_config(page_title="TENSOR-FORGE // EMPOWERAI", layout="wide", page_icon="⬛")

st.markdown("""
    <style>
    /* Global Command Center Theme */
    .stApp { background-color: #0a0a0a; color: #0f0; font-family: 'Courier New', Courier, monospace; }
    
    /* --- SIDEBAR & UPLOADER FIXES --- */
    [data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #333; }
    [data-testid="stHeader"] { background-color: #0a0a0a !important; }
    [data-testid="stFileUploadDropzone"] { background-color: #111 !important; border: 1px dashed #0f0 !important; }
    
    h1, h2, h3, h4, p, span, div, label { color: #0f0 !important; font-family: 'Courier New', Courier, monospace !important; }
    .stButton>button { background-color: #111; color: #00f0ff !important; border: 1px solid #00f0ff; box-shadow: 0 0 10px rgba(0,240,255,0.5); border-radius: 0px; transition: all 0.3s; }
    .stButton>button:hover { background-color: #00f0ff; color: #000 !important; box-shadow: 0 0 20px #00f0ff; }
    .stSlider > div > div > div > div { background-color: #00f0ff !important; }
    /* Headers */
    .neon-title { text-shadow: 0 0 10px #0f0; text-align: center; font-size: 2.5em; margin-bottom: 0px; font-weight: bold;}
    .neon-subtitle { color: #666 !important; text-align: center; margin-bottom: 30px; font-size: 0.9em; }
    hr { border-color: #333; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='neon-title'>TENSOR-FORGE // CMTF ENGINE</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-subtitle'>COUPLED FACTORIZATION FOR INFORMATION-MAXIMIZING DISCOVERY | STATUS: ONLINE</div>", unsafe_allow_html=True)

# --- 2. THE DEMO DATA GENERATOR ---
def generate_demo_tensor():
    # Entities: NVIDIA (AI), CEG (Nuclear Power), FCX (Copper), Apple, Ford
    entities = ['NVDA', 'CEG', 'FCX', 'AAPL', 'F']
    metrics = ['Trading Volume', 'Volatility', 'Options Flow']
    days = ['Day -4', 'Day -3', 'Day -2', 'Day -1', 'Today']
    
    # Initialize a random low-noise tensor (5 entities x 3 metrics x 5 days)
    np.random.seed(42)
    tensor = np.random.rand(5, 3, 5) * 0.1 
    
    # INJECT THE HIDDEN SIGNAL: 
    # AI boom (NVDA) causes Power shortage (CEG) requiring Copper (FCX)
    # They all spike together on 'Trading Volume' (Index 0) starting 'Day -2'
    for e_idx in [0, 1, 2]: # NVDA, CEG, FCX
        for t_idx in [2, 3, 4]: # Last 3 days
            tensor[e_idx, 0, t_idx] += 2.5 # Massive volume spike
            tensor[e_idx, 1, t_idx] += 1.5 # Volatility spike
            
    return tensor, entities, metrics, days

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.markdown("### UPLINK CONTROLS")
demo_mode = st.sidebar.button("[ EXECUTE STOCK MARKET DEMO ]")
st.sidebar.markdown("---")
st.sidebar.markdown("### MANUAL OVERRIDE")
file1 = st.sidebar.file_uploader("UPLOAD DATASET ALPHA (CSV)", type="csv")
file2 = st.sidebar.file_uploader("UPLOAD DATASET BETA (CSV)", type="csv")
rank = st.sidebar.slider("TARGET RANK (LATENT VECTORS)", 1, 5, 3)
start_manual = st.sidebar.button("[ INITIATE MANUAL ANALYSIS ]")

# Helper for dark Plotly charts
def dark_bar_chart(x, y, title, color):
    fig = px.bar(x=x, y=y, title=title)
    fig.update_traces(marker_color=color)
    fig.update_layout(
        plot_bgcolor='#111', paper_bgcolor='#0a0a0a', 
        font_color='#0f0', title_font_color='#00f0ff',
        xaxis=dict(showgrid=False, title=""), yaxis=dict(showgrid=True, gridcolor='#333', title="")
    )
    return fig

# --- 4. ENGINE EXECUTION ---
if demo_mode:
    with st.spinner("INITIALIZING ALTERNATING LEAST SQUARES (ALS) ALGORITHM..."):
        time.sleep(1.5) # Theatrical delay for the "processing" feel
        
        # Load and decompose
        X, entities, metrics, days = generate_demo_tensor()
        tensor_tl = tl.tensor(X)
        weights, factors = parafac(tensor_tl, rank=rank, init='random', tol=10e-6)
        
        st.success("TENSOR FACTORIZATION COMPLETE. HIDDEN CORRELATIONS EXTRACTED.")
        st.markdown("---")
        
        # --- 5. VISUALIZATION ---
        st.markdown("### [ LATENT PATTERN 1: THE SUPPLY-CHAIN CASCADE ]")
        st.markdown("Math has detected a highly correlated anomaly spanning **Tech, Utilities, and Materials**.")
        
        col1, col2, col3 = st.columns(3)
        
        # Plot Entity Vector
        with col1:
            fig_entity = dark_bar_chart(entities, factors[0][:, 0], "SHARED ENTITY VECTOR", "#00f0ff")
            st.plotly_chart(fig_entity, use_container_width=True)
            
        # Plot Metric Vector
        with col2:
            fig_metric = dark_bar_chart(metrics, factors[1][:, 0], "MARKET METRIC VECTOR", "#0f0")
            st.plotly_chart(fig_metric, use_container_width=True)
            
        # Plot Time Vector
        with col3:
            fig_time = dark_bar_chart(days, factors[2][:, 0], "TEMPORAL VECTOR", "#ff003c")
            st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("> **ANALYTICS ENGINE OUTPUT:** The model isolated a mathematically linked event where NVIDIA (AI), Constellation Energy (Nuclear), and Freeport-McMoRan (Copper) experienced simultaneous heavy trading volume over the last 72 hours, completely invisible to traditional sector-isolated analysis.")

elif file1 and file2:
    with st.spinner("UPLINK SECURED. ANALYZING DATA STRUCTURES..."):
        # 1. Read the CSVs
        df1 = pd.read_csv(file1, index_col=0)
        df2 = pd.read_csv(file2, index_col=0)
        
        # DECISION LOGIC: Are the files identical in shape for a 3D Tensor?
        if df1.shape == df2.shape and all(df1.index == df2.index):
            st.info("DETECTED SYMMETRIC DATA: EXECUTING FULL TENSOR STACK.")
            entities = df1.index.tolist()
            metrics_1 = df1.columns.tolist()
            
            # Stack into a 3D Tensor (Entities x Datasets x Features)
            X = np.stack([df1.values, df2.values], axis=1)
            tensor_tl = tl.tensor(X)
            
            # Run CP Factorization
            weights, factors = parafac(tensor_tl, rank=rank, init='random', tol=10e-6)
            
            st.success("MANUAL TENSOR FACTORIZATION COMPLETE.")
            st.markdown("---")
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig_entity = dark_bar_chart(entities, factors[0][:, 0], "EXTRACTED: SHARED ENTITY VECTOR", "#00f0ff")
                st.plotly_chart(fig_entity, use_container_width=True)
                
            with col2:
                fig_metric = dark_bar_chart(metrics_1, factors[2][:, 0], "EXTRACTED: FEATURE WEIGHTS", "#ff003c")
                st.plotly_chart(fig_metric, use_container_width=True)
                
        else:
            st.info("DETECTED DISPARATE DATA: EXECUTING ALIGNMENT & CROSS-DOMAIN FACTORIZATION.")
            # 1. Align by Index
            common_entities = df1.index.intersection(df2.index).tolist()
            
            if len(common_entities) < 2:
                st.error("ERROR: No shared entities (Row IDs) found between these files.")
            else:
                # Filter both to only include shared entities and ensure numeric data types
                df1_aligned = df1.loc[common_entities].select_dtypes(include=[np.number])
                df2_aligned = df2.loc[common_entities].select_dtypes(include=[np.number])
                
                # 2. FEATURE NORMALIZATION (Critical for disparate scales)
                # Ensure missing or zero-variance columns do not cause division by zero
                df1_std = df1_aligned.std().replace(0, 1)
                df2_std = df2_aligned.std().replace(0, 1)
                
                df1_norm = (df1_aligned - df1_aligned.mean()) / df1_std
                df2_norm = (df2_aligned - df2_aligned.mean()) / df2_std
                
                # 3. COUPLED REPRESENTATION
                # Concatenate on the feature axis to find the shared latent space across both domains.
                combined_matrix = np.hstack([df1_norm.values, df2_norm.values])
                
                # Perform SVD as a proxy for the shared Factor in CMTF
                u, s, vh = np.linalg.svd(combined_matrix, full_matrices=False)
                
                st.success(f"ALIGNMENT SUCCESSFUL. ANALYZING {len(common_entities)} SHARED ENTITIES.")
                st.markdown("---")
                
                # 4. VISUALIZATION
                col1, col2 = st.columns(2)
                with col1:
                    # Top Latent Factor for Entities
                    fig_entity = dark_bar_chart(common_entities, u[:, 0], "SHARED LATENT ENTITY VECTOR", "#00f0ff")
                    st.plotly_chart(fig_entity, use_container_width=True)
                with col2:
                    # Combined Feature Weights
                    all_features = df1_aligned.columns.tolist() + df2_aligned.columns.tolist()
                    fig_feature = dark_bar_chart(all_features, vh[0, :], "CROSS-DOMAIN FEATURE CORRELATION", "#ff003c")
                    st.plotly_chart(fig_feature, use_container_width=True)