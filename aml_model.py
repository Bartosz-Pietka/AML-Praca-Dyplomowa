import pandas as pd
import numpy as np

df_tx = pd.read_csv('Transakcje.csv', sep=';', decimal=',')
df_risk = pd.read_csv('Country risk.csv', sep=';')

df_tx['USD_amount'] = pd.to_numeric(df_tx['USD_amount'], errors='coerce')

print(f"Status: Wczytano {len(df_tx)} rekordów.")
print(f"Format kwoty: {df_tx['USD_amount'].dtype}")

df_combined = pd.merge(df_tx, 
                       df_risk[['Bene_Country', 'Country_risk2']], 
                       on='Bene_Country', 
                       how='left')

df_combined['Country_risk2'] = df_combined['Country_risk2'].fillna('Medium')

risk_map = {'Low': 10, 'Medium': 45, 'High': 90}
df_combined['Country_Score'] = df_combined['Country_risk2'].map(risk_map)

mean_amt = df_combined['USD_amount'].mean()
std_amt = df_combined['USD_amount'].std()
df_combined['Amount_Score'] = ((df_combined['USD_amount'] - mean_amt) / std_amt).clip(0, 5) * 20

df_combined['Total_Risk_Score'] = (
    df_combined['Country_Score'] * 0.7 + 
    df_combined['Amount_Score'] * 0.2 +
    np.where(df_combined['Transaction_Type'] == 'QUICK-PAYMENT', 10, 0)
)

def classify_risk(score):
    if score > 70: return 'High Risk'
    if score > 40: return 'Medium Risk'
    return 'Low Risk'

df_combined['Final_Risk_Level'] = df_combined['Total_Risk_Score'].apply(classify_risk)

print(df_combined['Final_Risk_Level'].value_counts())

import matplotlib.pyplot as plt

risk_distribution = df_combined['Final_Risk_Level'].value_counts()

plt.figure(figsize=(10, 6))
risk_distribution.plot(kind='bar', color=['#2ecc71', '#f1c40f', '#e74c3c'])

plt.title('Rozkład poziomów ryzyka transakcyjnego (Model AML)', fontsize=14)
plt.xlabel('Kategoria ryzyka', fontsize=12)
plt.ylabel('Liczba transakcji', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(risk_distribution):
    plt.text(i, v + 500, str(v), ha='center', fontweight='bold')

plt.show()

top_alerts = df_combined.sort_values(by='Total_Risk_Score', ascending=False).head(10)

top_alerts[['Transaction_Id', 'Sender_Country', 'Bene_Country', 'USD_amount', 'Total_Risk_Score']]
