import os
import streamlit as st
from dotenv import load_dotenv
from asset_engine import RWAContentEngine

# Configuration
st.set_page_config(
    page_title="PROP",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Newspaper Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=EB+Garamond:ital,wght@0,400;0,700;1,400&display=swap');

    /* Background and Base Font */
    .stApp {
        background-color: #f4f1ea !important;
        color: #1a1a1a !important;
        font-family: 'EB Garamond', serif !important;
    }

    /* Masthead */
    .masthead {
        font-family: 'Playfair Display', serif;
        font-size: 120px;
        text-align: center;
        border-bottom: 4px double #1a1a1a;
        margin-top: -50px;
        margin-bottom: 5px;
        letter-spacing: -4px;
        line-height: 1;
    }
    
    .sub-masthead {
        text-align: center;
        border-bottom: 2px solid #1a1a1a;
        margin-bottom: 40px;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 4px;
        padding-bottom: 8px;
        font-weight: 700;
    }

    /* Headlines */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #000000 !important;
        line-height: 1.1 !important;
        text-align: center;
    }
    
    .headline {
        font-family: 'Playfair Display', serif;
        font-size: 56px;
        font-weight: 700;
        margin-bottom: 30px;
        text-align: center;
        border-bottom: 1px solid #1a1a1a;
        padding-bottom: 10px;
    }

    /* Multi-column layout for Fact Sheet */
    .newspaper-columns {
        column-count: 3;
        column-gap: 50px;
        column-rule: 1px solid #1a1a1a;
        text-align: justify;
        font-size: 18px;
        line-height: 1.4;
    }
    
    .newspaper-columns p {
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 0px !important;
        border: 1px solid #1a1a1a !important;
        background-color: transparent !important;
        color: #1a1a1a !important;
        font-family: 'EB Garamond', serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700 !important;
    }
    
    .stButton > button:hover {
        background-color: #1a1a1a !important;
        color: #f4f1ea !important;
        border: 1px solid #1a1a1a !important;
    }

    /* Input Field */
    .stTextInput > div > div > input {
        border-radius: 0px !important;
        border: none !important;
        border-bottom: 2px solid #1a1a1a !important;
        background-color: transparent !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 32px !important;
        text-align: center;
        color: #1a1a1a !important;
    }

    /* Loading Overlay */
    .loading-text {
        font-family: 'EB Garamond', serif;
        font-size: 28px;
        font-style: italic;
        text-align: center;
        margin-top: 50px;
        color: #1a1a1a;
    }
    
    /* Separator */
    hr {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 40px 0;
    }

    /* Remove Streamlit default elements */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    
    /* Page Container */
    .page-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Init
if 'page' not in st.session_state:
    st.session_state.page = 0  # 0: Input, 1: Fact Sheet, 2: Public Summary, 3: Risk Assessment
if 'results' not in st.session_state:
    st.session_state.results = None

def main():
    # Masthead
    st.markdown('<div class="masthead">PROP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-masthead">Journal of Asset Tokenization & Digital Finance</div>', unsafe_allow_html=True)

    if st.session_state.page == 0:
        # Page 0: Input
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            topic = st.text_input("Enter Asset Topic", placeholder="Enter Asset Topic", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col2:
                if st.button("Generate Report"):
                    if topic:
                        load_dotenv()
                        api_key = os.getenv("GEMINI_API_KEY", "").strip()
                        if api_key:
                            loading_placeholder = st.empty()
                            loading_placeholder.markdown('<div class="loading-text">Typesetting Report...</div>', unsafe_allow_html=True)
                            try:
                                engine = RWAContentEngine(api_key=api_key)
                                fact_sheet = engine.generate_article(topic)
                                public_summary = engine.convert_to_linkedin(fact_sheet)
                                risk_assessment = engine.generate_summary(fact_sheet)
                                
                                st.session_state.results = {
                                    "fact_sheet": fact_sheet,
                                    "public_summary": public_summary,
                                    "risk_assessment": risk_assessment,
                                    "topic": topic
                                }
                                st.session_state.page = 1
                                loading_placeholder.empty()
                                st.rerun()
                            except Exception as e:
                                loading_placeholder.empty()
                                st.error(f"The system is busy. Please wait 60 seconds.")
                        else:
                            st.error("Missing GEMINI_API_KEY")
                    else:
                        st.warning("Please enter a topic.")

    elif st.session_state.page >= 1:
        res = st.session_state.results
        
        # Navigation Top
        st.markdown('<div class="page-container">', unsafe_allow_html=True)
        ncol1, ncol2, ncol3 = st.columns([1, 8, 1])
        with ncol1:
            if st.button("← Reset"):
                st.session_state.page = 0
                st.session_state.results = None
                st.rerun()
        
        # Page Content
        if st.session_state.page == 1:
            st.markdown('<div class="headline">I. THE FACT SHEET: CORE ANALYSIS</div>', unsafe_allow_html=True)
            # Use columns for broadsheet effect
            st.markdown(f'<div class="newspaper-columns">{res["fact_sheet"]}</div>', unsafe_allow_html=True)
            
        elif st.session_state.page == 2:
            st.markdown('<div class="headline">II. MARKET NEWS: PUBLIC SUMMARY</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="max-width: 900px; margin: 0 auto; font-size: 20px; line-height: 1.6; text-align: justify;">{res["public_summary"]}</div>', unsafe_allow_html=True)
            
        elif st.session_state.page == 3:
            st.markdown('<div class="headline">III. THE LEDGER: RISK ASSESSMENT</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="max-width: 900px; margin: 0 auto; font-size: 18px; line-height: 1.5; font-family: monospace; background: rgba(0,0,0,0.05); padding: 30px; border: 1px solid #1a1a1a;">{res["risk_assessment"]}</div>', unsafe_allow_html=True)

        # Pagination Footer
        st.markdown("<hr>", unsafe_allow_html=True)
        fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
        with fcol1:
            if st.session_state.page > 1:
                if st.button("← Previous Page"):
                    st.session_state.page -= 1
                    st.rerun()
        with fcol2:
            st.markdown(f"<p style='text-align: center; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;'>Sheet {st.session_state.page} / 3</p>", unsafe_allow_html=True)
        with fcol3:
            if st.session_state.page < 3:
                if st.button("Next Page →"):
                    st.session_state.page += 1
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
