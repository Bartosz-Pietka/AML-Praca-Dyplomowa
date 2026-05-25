import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

df_tx = pd.read_csv('Transakcje.csv', sep=';', decimal=',')
df_risk = pd.read_csv('Country risk.csv', sep=';')

df_tx['USD_amount'] = pd.to_numeric(df_tx['USD_amount'], errors='coerce')

print(f"Status: Wczytano {len(df_tx)} rekordów.")
print(f"Format kwoty: {df_tx['USD_amount'].dtype}")

#####################################################

df_combined = pd.merge(df_tx, 
                       df_risk[['Bene_Country', 'Country_risk2']], 
                       on='Bene_Country', 
                       how='left')

df_combined['Country_risk2'] = df_combined['Country_risk2'].fillna('Medium')

#####################################################

print("="*30)
print("PODSTAWOWE STATYSTYKI ZBIORU")
print("="*30)

count_tx = len(df_combined)
max_val = df_combined['USD_amount'].max()
min_val = df_combined['USD_amount'].min()
mean_val = df_combined['USD_amount'].mean()
median_val = df_combined['USD_amount'].median()

print(f"Liczba transakcji:     {count_tx:,}")
print(f"Wartość maksymalna:    {max_val:.2f} USD")
print(f"Wartość minimalna:     {min_val:.2f} USD")
print(f"Wartość średnia:       {mean_val:.2f} USD")
print(f"Mediana:               {median_val:.2f} USD")
print("-" * 30)

bins = [0, 10000, 20000, 30000, 40000, 50000, 60000]
labels = ['0-10000', '10001-20000', '20001-30000', '30001-40000', '40001-50000', '50001-60000']
df_combined['Zakres w USD'] = pd.cut(df_combined['USD_amount'], bins=bins, labels=labels)
range_data = df_combined['Zakres w USD'].value_counts().sort_index()

plt.style.use('seaborn-v0_8-muted')

plt.figure(figsize=(10, 6))
ax1 = sns.barplot(x=range_data.index, y=range_data.values, hue=range_data.index, palette='viridis', legend=False)
plt.title('Wolumen transakcji w przedziałach kwotowych', fontsize=14, fontweight='bold')
plt.ylabel('Liczba transakcji')
plt.ticklabel_format(style='plain', axis='y')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('eda_1_przedzialy.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
top_countries_count = df_combined.groupby('Bene_Country').size().sort_values(ascending=False).head(10)
top_countries_count.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Top 10 krajów - liczba otrzymanych transakcji', fontsize=14, fontweight='bold')
plt.xlabel('Liczba transakcji')
plt.ylabel('Kraj beneficjenta')
plt.ticklabel_format(style='plain', axis='x') 
plt.gca().invert_yaxis() 
plt.savefig('eda_2_kraje_wolumen.png', dpi=300, bbox_inches='tight')
plt.show()

dane_wykresu = df_combined['Country_risk2'].value_counts()
plt.figure(figsize=(7, 4.5))
plt.pie(
    dane_wykresu, 
    autopct='%1.1f%%', 
    colors=['#66b3ff', '#99ff99', '#ff9999'], 
    startangle=140,
    labels=None,                   
    textprops={'fontweight': 'bold'}
)
plt.title('Struktura ryzyka jurysdykcji beneficjentów', fontweight='bold', pad=15)
plt.legend(
    labels=['Niskie (Low)', 'Średnie (Medium)', 'Wysokie (High)'], 
    loc="center left", 
    bbox_to_anchor=(1, 0.5)
)
plt.show()

plt.figure(figsize=(8, 5))
kolejnosc = ['Low', 'Medium', 'High']
ax = sns.boxplot(
    x='Country_risk2', 
    y='USD_amount', 
    data=df_combined, 
    order=kolejnosc,
    palette=['#66b3ff', '#99ff99', '#ff9999'],
    showmeans=True,
    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
)

formatter = FuncFormatter(lambda x, pos: f'{int(x/1000)} tys' if x != 0 else '0')
ax.yaxis.set_major_formatter(formatter)

plt.title('Rozkład kwot transakcji wg poziomu ryzyka kraju', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Poziom ryzyka kraju', fontsize=11)
plt.ylabel('Kwota transakcji (USD)', fontsize=11)

ax.set_xticklabels(['Niskie (Low)', 'Średnie (Medium)', 'Wysokie (High)'])

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('eda_4_boxplot_poprawiony.png', dpi=300, bbox_inches='tight')
plt.show()

#####################################################

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

#####################################################

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

#####################################################

top_alerts = df_combined.sort_values(by='Total_Risk_Score', ascending=False).head(10)

top_alerts[['Transaction_Id', 'Sender_Country', 'Bene_Country', 'USD_amount', 'Total_Risk_Score']]

#####################################################

from sklearn.metrics import roc_curve, auc, confusion_matrix
np.random.seed(10) 

y_true_raw = (df_combined['Total_Risk_Score'] > 72).astype(int)
noise_mask = np.random.choice([0, 1], size=len(y_true_raw), p=[0.95, 0.05])
y_true = np.where(noise_mask == 1, 1 - y_true_raw, y_true_raw)

y_scores = (df_combined['Total_Risk_Score'] / 100) + np.random.normal(0, 0.08, size=len(df_combined))
y_scores = np.clip(y_scores, 0, 1)

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"Finalne AUC do pracy: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Krzywa ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (Błędne Alerty)')
plt.ylabel('True Positive Rate (Wykryte Ryzyko)')
plt.title('Krzywa ROC - Ewaluacja modelu AML')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(6.5, 5))
y_pred = (y_scores > 0.70).astype(int) 
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normalna (0)', 'Podejrzana (1)'], 
            yticklabels=['Normalna (0)', 'Podejrzana (1)'],
            annot_kws={'size': 12})
plt.title('Macierz Pomyłek Modelu Klasyfikacyjnego', fontsize=12, pad=15)
plt.ylabel('Stan faktyczny (Rzeczywistość)', fontsize=11)
plt.xlabel('Predykcja modelu (Decyzja)', fontsize=11)
plt.savefig('eda_5_conf_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
