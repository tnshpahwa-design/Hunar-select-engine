import streamlit as st

st.set_page_config(page_title="Hunar Select Recruitment Intelligence Engine", layout="wide")

st.title("Hunar Select: Recruitment Intelligence Engine")
st.caption("Pick an industry and company size. The system finds target companies, hiring signals and decision-makers.")

col1, col2, col3 = st.columns(3)
industry = col1.selectbox("Target industry", ["Quick Commerce", "Retail", "Logistics", "BFSI", "Healthcare"])
size = col2.selectbox("Company size (employees)", ["1,000 to 5,000", "5,000 to 20,000", "20,000+"])
col3.text_input("Geography", value="India", disabled=True)

if st.button("Run research", type="primary"):
    st.info(f"Research for {industry}, {size}, India will run here from Stage 3.")
