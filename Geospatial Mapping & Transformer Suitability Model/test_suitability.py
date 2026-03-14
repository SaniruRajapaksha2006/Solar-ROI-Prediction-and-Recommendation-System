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

print("Calling assessment API...")
response = requests.post(f'{API}/api/assess', json=payload)
data     = response.json()

if 'error' in data:
    print(f"API Error: {data['error']}")
    exit()

results = data['transformers']
print(f"Got {len(results)} transformers")

y_true = [1 if tf['canSupport'] else 0 for tf in results]
y_pred = [1 if tf['score'] >= 60 else 0 for tf in results]

print("\nGround truth (canSupport) :", y_true)
print("Predicted   (score >= 60) :", y_pred)