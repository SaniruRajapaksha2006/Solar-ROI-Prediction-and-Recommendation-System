import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

API = 'http://localhost:5000'

payload = {
    'latitude'     : 6.849,
    'longitude'    : 79.9247,
    'solarCapacity': 5.0,
    'searchRadius' : 5000
}