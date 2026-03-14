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

print("\n" + "=" * 55)
print("    CLASSIFICATION REPORT — SUITABILITY SCORING")
print("=" * 55)
print(classification_report(
    y_true, y_pred,
    target_names=['Not Suitable', 'Suitable'],
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Suitable', 'Suitable'],
    yticklabels=['Not Suitable', 'Suitable'],
    linewidths=0.5,
    linecolor='gray'
)
plt.title('Confusion Matrix — Transformer Suitability Scoring', fontsize=13, pad=15)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('Actual Label (canSupport)', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix_suitability.png', dpi=150)
plt.show()
print("Saved → confusion_matrix_suitability.png")
