# styles.py 
import streamlit as st
def inject_global_styles():
    """Styles CSS identiques partout"""
    st.markdown("""
    <style>
        /* TOUS TES STYLES CSS  */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
        }
        [data-testid="stSidebar"] * { color: white !important; }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 15px; text-align: center;
            color: white; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .metric-card { /* tes cards */ }
        .workflow-step { /* tes steps */ }
        .step-number { /* tes numéros */ }
        /* TOUS LES AUTRES STYLES... */
    </style>
    """, unsafe_allow_html=True)

def page_header(title, subtitle=""):
    """Header cohérent partout"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
