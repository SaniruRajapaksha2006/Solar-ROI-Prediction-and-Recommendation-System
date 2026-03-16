import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic

st.set_page_config(
    page_title="SolarGrid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)