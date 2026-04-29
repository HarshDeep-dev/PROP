PROP
Institutional Intelligence Terminal

PROP is a high-definition utility for Real-World Asset (RWA) tokenization. It converts raw asset data into structured, newspaper-style financial reports for institutional analysis.

Features
Fact Sheet: Technical and regulatory asset analysis.

Market News: Direct, hype-free public summaries.

Risk Ledger: Bulleted risk vs. reward assessment.

Resilient Engine: Automated rate-limit handling and error recovery.

Tech Stack
Language: Python 3.12

AI Engine: Google Gemini 2.5 Flash-Lite

Interface: Streamlit (Custom Serif/Newspaper UI)

Setup
Clone the Repository:

Bash
git clone https://github.com/YOUR_USERNAME/PROP.git
cd PROP

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables:**
    Create a `.env` file and add your API key:
    ```text
    GEMINI_API_KEY=your_actual_key_here
    ```
4.  **Run the Terminal:**
    ```bash
    streamlit run app.py
    ```
