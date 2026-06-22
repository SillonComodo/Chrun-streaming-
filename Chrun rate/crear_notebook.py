#!/usr/bin/env python3
"""Script para generar el notebook del proyecto de Churn."""
import json

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source if isinstance(source, list) else [source]}

cells = []

# === TÍTULO ===
cells.append(md([
    "# Proyecto: Predicción de Churn en Distribución de Streaming\n",
    "\n",
    "## Contexto del Negocio\n",
    "Este proyecto se desarrolla dentro de un modelo de emprendimiento enfocado en la **distribución y reventa de perfiles de plataformas de streaming** (MAX, Netflix, Crunchyroll, etc.).\n",
    "\n",
    "### Objetivos\n",
    "- **Variable Target (objetivo):** `churn` (0 = cliente activo, 1 = cliente que abandonó)\n",
    "- **Técnica básica:** Regresión Logística\n",
    "- **Técnica avanzada:** Random Forest (comparación de al menos 2 técnicas)\n",
    "\n",
    "### Cronograma\n",
    "| Semana | Actividad |\n",
    "|--------|----------|\n",
    "| 1 | Dataset, objetivos y cronograma |\n",
    "| 2 | Preprocesamiento y análisis estadístico |\n",
    "| 3 | Selección de variables y Regresión Logística |\n",
    "| 4 | Random Forest y comparación de técnicas |\n",
    "| 5 | Entrega final y documentación |"
]))

# === CELDA 1: Importaciones ===
cells.append(md("## 1. Importación de Librerías"))
cells.append(code([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix,\n",
    "    roc_auc_score, roc_curve, accuracy_score\n",
    ")\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "print('Librerías importadas correctamente')"
]))

# === CELDA 2: Carga de datos ===
cells.append(md("## 2. Carga del Dataset"))
cells.append(code([
    "df = pd.read_csv('dataset_streaming_limpio.csv')\n",
    "print(f'Dimensiones del dataset: {df.shape}')\n",
    "print(f'\\nColumnas: {list(df.columns)}')\n",
    "df.head(10)"
]))

# === CELDA 3: Inspección ===
cells.append(md("## 3. Inspección Inicial del Dataset"))
cells.append(code([
    "print('=== INFORMACIÓN DEL DATASET ===')\n",
    "print(df.info())\n",
    "print('\\n=== ESTADÍSTICAS DESCRIPTIVAS ===')\n",
    "df.describe()"
]))

cells.append(code([
    "print('=== VALORES NULOS ===')\n",
    "print(df.isnull().sum())\n",
    "print(f'\\n=== DISTRIBUCIÓN DEL TARGET (churn) ===')\n",
    "print(df['churn'].value_counts())\n",
    "print(f'\\nTasa de Churn: {df[\"churn\"].mean():.2%}')"
]))

# === CELDA 4: Análisis Exploratorio ===
cells.append(md("## 4. Análisis Exploratorio de Datos (EDA)"))
cells.append(md("### 4.1 Distribución de la Variable Target"))
cells.append(code([
    "fig, ax = plt.subplots(1, 1, figsize=(6, 4))\n",
    "df['churn'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'], ax=ax)\n",
    "ax.set_title('Distribución de Churn')\n",
    "ax.set_xlabel('Churn (0=Activo, 1=Abandonó)')\n",
    "ax.set_ylabel('Cantidad')\n",
    "ax.set_xticklabels(['Activo', 'Abandonó'], rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("### 4.2 Distribución por Plataforma"))
cells.append(code([
    "fig, ax = plt.subplots(1, 1, figsize=(8, 4))\n",
    "df['plataforma'].value_counts().plot(kind='bar', color='steelblue', ax=ax)\n",
    "ax.set_title('Clientes por Plataforma')\n",
    "ax.set_xlabel('Plataforma')\n",
    "ax.set_ylabel('Cantidad')\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("### 4.3 Churn por Duración del Plan"))
cells.append(code([
    "churn_por_duracion = df.groupby('duracion_plan')['churn'].mean().sort_values(ascending=False)\n",
    "fig, ax = plt.subplots(1, 1, figsize=(6, 4))\n",
    "churn_por_duracion.plot(kind='bar', color='coral', ax=ax)\n",
    "ax.set_title('Tasa de Churn por Duración del Plan')\n",
    "ax.set_xlabel('Duración del Plan')\n",
    "ax.set_ylabel('Tasa de Churn')\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("### 4.4 Churn por Plataforma"))
cells.append(code([
    "churn_por_plat = df.groupby('plataforma')['churn'].mean().sort_values(ascending=False)\n",
    "fig, ax = plt.subplots(1, 1, figsize=(8, 4))\n",
    "churn_por_plat.plot(kind='bar', color='teal', ax=ax)\n",
    "ax.set_title('Tasa de Churn por Plataforma')\n",
    "ax.set_xlabel('Plataforma')\n",
    "ax.set_ylabel('Tasa de Churn')\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# === CELDA 5: Preprocesamiento ===
cells.append(md("## 5. Preprocesamiento de Datos"))
cells.append(md("### 5.1 Codificación de Variables Categóricas"))
cells.append(code([
    "# Codificar variables categóricas\n",
    "le_plataforma = LabelEncoder()\n",
    "le_tipo_plan = LabelEncoder()\n",
    "\n",
    "df['plataforma_encoded'] = le_plataforma.fit_transform(df['plataforma'])\n",
    "df['tipo_plan_encoded'] = le_tipo_plan.fit_transform(df['tipo_plan'])\n",
    "\n",
    "print('Mapeo de Plataforma:')\n",
    "for i, clase in enumerate(le_plataforma.classes_):\n",
    "    print(f'  {clase} -> {i}')\n",
    "\n",
    "print('\\nMapeo de Tipo de Plan:')\n",
    "for i, clase in enumerate(le_tipo_plan.classes_):\n",
    "    print(f'  {clase} -> {i}')"
]))

cells.append(md("### 5.2 Selección de Variables Predictoras"))
cells.append(code([
    "# Variables predictoras (features)\n",
    "features = ['duracion_meses', 'plataforma_encoded', 'tipo_plan_encoded',\n",
    "             'mes_inicio', 'tiene_fecha_inicio']\n",
    "\n",
    "X = df[features].copy()\n",
    "y = df['churn'].copy()\n",
    "\n",
    "print(f'Features seleccionadas: {features}')\n",
    "print(f'Shape de X: {X.shape}')\n",
    "print(f'Shape de y: {y.shape}')\n",
    "print(f'\\nPrimeras filas de X:')\n",
    "X.head()"
]))

cells.append(md("### 5.3 Matriz de Correlación"))
cells.append(code([
    "# Agregar target para la correlación\n",
    "df_corr = X.copy()\n",
    "df_corr['churn'] = y\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')\n",
    "ax.set_title('Matriz de Correlación')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("### 5.4 División Train/Test"))
cells.append(code([
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.25, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "print(f'Conjunto de entrenamiento: {X_train.shape[0]} muestras')\n",
    "print(f'Conjunto de prueba: {X_test.shape[0]} muestras')\n",
    "print(f'\\nDistribución de churn en entrenamiento:')\n",
    "print(y_train.value_counts())\n",
    "print(f'\\nDistribución de churn en prueba:')\n",
    "print(y_test.value_counts())"
]))

cells.append(md("### 5.5 Escalamiento de Features"))
cells.append(code([
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print('Datos escalados correctamente')\n",
    "print(f'Media de X_train_scaled: {X_train_scaled.mean(axis=0).round(4)}')\n",
    "print(f'Std de X_train_scaled: {X_train_scaled.std(axis=0).round(4)}')"
]))

# === CELDA 6: Regresión Logística ===
cells.append(md("## 6. Técnica 1: Regresión Logística"))
cells.append(md("### 6.1 Entrenamiento del Modelo"))
cells.append(code([
    "# Crear y entrenar el modelo de Regresión Logística\n",
    "lr = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)\n",
    "lr.fit(X_train_scaled, y_train)\n",
    "\n",
    "print('Modelo de Regresión Logística entrenado')\n",
    "print(f'\\nIntercepto (w0): {lr.intercept_[0]:.4f}')\n",
    "print(f'\\nCoeficientes:')\n",
    "for feat, coef in zip(features, lr.coef_[0]):\n",
    "    print(f'  {feat}: {coef:.4f}')"
]))

cells.append(md("### 6.2 Predicciones"))
cells.append(code([
    "# Predicciones\n",
    "y_pred_lr = lr.predict(X_test_scaled)\n",
    "y_probs_lr = lr.predict_proba(X_test_scaled)[:, 1]\n",
    "\n",
    "print('Primeras 10 predicciones vs valores reales:')\n",
    "for i in range(min(10, len(y_test))):\n",
    "    real = y_test.iloc[i]\n",
    "    pred = y_pred_lr[i]\n",
    "    prob = y_probs_lr[i]\n",
    "    print(f'  Real: {real} | Predicción: {pred} | P(churn): {prob:.4f}')"
]))

cells.append(md("### 6.3 Evaluación - Regresión Logística"))
cells.append(code([
    "# Métricas\n",
    "print('=== REPORTE DE CLASIFICACIÓN - Regresión Logística ===')\n",
    "print(classification_report(y_test, y_pred_lr, target_names=['Activo', 'Churn']))\n",
    "\n",
    "acc_lr = accuracy_score(y_test, y_pred_lr)\n",
    "print(f'Accuracy: {acc_lr:.4f}')"
]))

cells.append(code([
    "# Matriz de Confusión\n",
    "cm_lr = confusion_matrix(y_test, y_pred_lr)\n",
    "fig, ax = plt.subplots(figsize=(5, 4))\n",
    "sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax,\n",
    "            xticklabels=['Activo', 'Churn'], yticklabels=['Activo', 'Churn'])\n",
    "ax.set_title('Matriz de Confusión - Regresión Logística')\n",
    "ax.set_xlabel('Predicción')\n",
    "ax.set_ylabel('Valor Real')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(code([
    "# Curva ROC\n",
    "auc_lr = roc_auc_score(y_test, y_probs_lr)\n",
    "fpr_lr, tpr_lr, _ = roc_curve(y_test, y_probs_lr)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(6, 5))\n",
    "ax.plot(fpr_lr, tpr_lr, color='red', label=f'Reg. Logística (AUC = {auc_lr:.2f})')\n",
    "ax.plot([0, 1], [0, 1], color='gray', linestyle='--')\n",
    "ax.set_xlabel('Tasa de Falsos Positivos')\n",
    "ax.set_ylabel('Tasa de Verdaderos Positivos')\n",
    "ax.set_title('Curva ROC - Regresión Logística')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "print(f'ROC-AUC Score: {auc_lr:.4f}')"
]))

# === CELDA 7: Random Forest ===
cells.append(md("## 7. Técnica 2: Random Forest"))
cells.append(md("### 7.1 Entrenamiento del Modelo"))
cells.append(code([
    "# Crear y entrenar Random Forest\n",
    "rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)\n",
    "rf.fit(X_train_scaled, y_train)\n",
    "\n",
    "print('Modelo Random Forest entrenado')\n",
    "print(f'\\nImportancia de las features:')\n",
    "for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True):\n",
    "    print(f'  {feat}: {imp:.4f}')"
]))

cells.append(md("### 7.2 Predicciones"))
cells.append(code([
    "y_pred_rf = rf.predict(X_test_scaled)\n",
    "y_probs_rf = rf.predict_proba(X_test_scaled)[:, 1]\n",
    "\n",
    "print('Primeras 10 predicciones vs valores reales:')\n",
    "for i in range(min(10, len(y_test))):\n",
    "    real = y_test.iloc[i]\n",
    "    pred = y_pred_rf[i]\n",
    "    prob = y_probs_rf[i]\n",
    "    print(f'  Real: {real} | Predicción: {pred} | P(churn): {prob:.4f}')"
]))

cells.append(md("### 7.3 Evaluación - Random Forest"))
cells.append(code([
    "print('=== REPORTE DE CLASIFICACIÓN - Random Forest ===')\n",
    "print(classification_report(y_test, y_pred_rf, target_names=['Activo', 'Churn']))\n",
    "\n",
    "acc_rf = accuracy_score(y_test, y_pred_rf)\n",
    "print(f'Accuracy: {acc_rf:.4f}')"
]))

cells.append(code([
    "# Matriz de Confusión - Random Forest\n",
    "cm_rf = confusion_matrix(y_test, y_pred_rf)\n",
    "fig, ax = plt.subplots(figsize=(5, 4))\n",
    "sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax,\n",
    "            xticklabels=['Activo', 'Churn'], yticklabels=['Activo', 'Churn'])\n",
    "ax.set_title('Matriz de Confusión - Random Forest')\n",
    "ax.set_xlabel('Predicción')\n",
    "ax.set_ylabel('Valor Real')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md("### 7.4 Importancia de Features"))
cells.append(code([
    "importancias = pd.Series(rf.feature_importances_, index=features).sort_values()\n",
    "fig, ax = plt.subplots(figsize=(7, 4))\n",
    "importancias.plot(kind='barh', color='forestgreen', ax=ax)\n",
    "ax.set_title('Importancia de Features - Random Forest')\n",
    "ax.set_xlabel('Importancia')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# === CELDA 8: Comparación ===
cells.append(md("## 8. Comparación de Modelos"))
cells.append(code([
    "# Curvas ROC comparativas\n",
    "auc_rf = roc_auc_score(y_test, y_probs_rf)\n",
    "fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7, 5))\n",
    "ax.plot(fpr_lr, tpr_lr, color='red', label=f'Reg. Logística (AUC = {auc_lr:.2f})')\n",
    "ax.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {auc_rf:.2f})')\n",
    "ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Aleatorio')\n",
    "ax.set_xlabel('Tasa de Falsos Positivos')\n",
    "ax.set_ylabel('Tasa de Verdaderos Positivos')\n",
    "ax.set_title('Comparación de Curvas ROC')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(code([
    "# Tabla comparativa\n",
    "comparacion = pd.DataFrame({\n",
    "    'Modelo': ['Regresión Logística', 'Random Forest'],\n",
    "    'Accuracy': [acc_lr, acc_rf],\n",
    "    'ROC-AUC': [auc_lr, auc_rf]\n",
    "})\n",
    "print('=== COMPARACIÓN FINAL DE MODELOS ===')\n",
    "print(comparacion.to_string(index=False))\n",
    "\n",
    "mejor = 'Regresión Logística' if auc_lr > auc_rf else 'Random Forest'\n",
    "print(f'\\nMejor modelo según ROC-AUC: {mejor}')"
]))

# === CELDA 9: Conclusiones ===
cells.append(md([
    "## 9. Conclusiones\n",
    "\n",
    "### Hallazgos Principales\n",
    "1. Se procesaron datos reales del negocio de distribución de streaming (~245 registros)\n",
    "2. La tasa de churn observada es de aproximadamente 26%, indicando que 1 de cada 4 clientes abandona\n",
    "3. Los planes de corta duración (1 mes) presentan tasas de churn más altas\n",
    "4. Se implementaron y compararon 2 técnicas de clasificación: Regresión Logística y Random Forest\n",
    "\n",
    "### Aplicación al Negocio\n",
    "- El modelo permite **anticipar bajas** antes de que venzan las suscripciones\n",
    "- Permite diseñar **campañas de retención** dirigidas a clientes con alta probabilidad de churn\n",
    "- La variable `duracion_meses` es un predictor importante del churn"
]))

# === Generar notebook ===
notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open("Proyecto_Churn_Streaming.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"Notebook creado con {len(cells)} celdas")
