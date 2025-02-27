#!/bin/bash
pip install -r requirements.txt  # Install dependencies
streamlit run app.py --server.port=8000 --server.address=0.0.0.0

