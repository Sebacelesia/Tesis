import streamlit as st
from typing import Optional
from io import BytesIO

def pdf_uploader(label: str = "Subí un PDF") -> Optional[BytesIO]:
    return st.file_uploader(label, type=["pdf"])
