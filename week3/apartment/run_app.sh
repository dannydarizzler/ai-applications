#!/bin/bash
# Run this script to start the Gradio app
# It will open a browser window automatically

cd "$(dirname "$0")"
echo "Starting Apartment Price Predictor..."
/opt/anaconda3/bin/python app.py
