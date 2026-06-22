#!/usr/bin/env python3
"""Reemplaza SOLO la sección de Random Forest del notebook del proyecto
   usando la metodología de la práctica de clase 3_Random_Forest_mejorada.ipynb"""
import json

RANDOM_STATE = 42

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source if isinstance(source, list) else [source]}

# ── Cargar notebook existente ──────────────────────────────────────────────────
with open("Proyecto_Churn_Streaming.ipynb") as f:
    nb = json.load(f)

# Las celdas 0-36 (título → Reg. Logística con ROC) se conservan intactas.
# Las celdas 37-46 (sección RF anterior) se sustituyen.
# Las celdas 47-50 (comparación y conclusiones) se conservan.

celdas_antes_rf = nb["cells"][:37]   # todo antes del bloque RF
celdas_despues_rf = nb["cells"][47:] # comparación y conclusiones

# ── Actualizar imports (celda 2) para incluir lo que necesita la práctica ─────
celdas_antes_rf[2]["source"] = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier, export_text\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix, ConfusionMatrixDisplay,\n",
    "    roc_auc_score, roc_curve, accuracy_score,\n",
    "    precision_score, recall_score, f1_score\n",
    ")\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "RANDOM_STATE = 42\n",
    "print('Librerías importadas correctamente')"
]

# ── Nuevas celdas de Random Forest (metodología de la práctica) ───────────────
nuevas_celdas_rf = []

# --- Título de sección ---
nuevas_celdas_rf.append(md([
    "## 7. Técnica 2: Random Forest\n",
    "\n",
    "Un **Random Forest** entrena varios árboles de decisión. Cada árbol se entrena "
    "con una muestra aleatoria de los datos y con subconjuntos aleatorios de variables. "
    "Finalmente, el bosque decide por **votación mayoritaria**. "
    "Esto produce modelos más robustos que un solo árbol de decisión.\n",
    "\n",
    "Seguimos la misma metodología vista en clase:\n",
    "1. Entrenamiento del modelo\n",
    "2. Evaluación con tabla train/test (accuracy, precision, recall, F1)\n",
    "3. Reporte de clasificación completo\n",
    "4. Matriz de confusión con `ConfusionMatrixDisplay`\n",
    "5. Importancia de variables\n",
    "6. Exploración interna: votación de los árboles para un caso\n",
    "7. Estadísticas de profundidad y hojas del bosque\n",
    "8. Comparación RF vs. mejor árbol del bosque vs. árbol independiente (GridSearchCV)\n",
    "9. Experimentación con hiperparámetros"
]))

# --- 7.1 Entrenamiento ---
nuevas_celdas_rf.append(md("### 7.1 Entrenamiento del Modelo"))
nuevas_celdas_rf.append(code([
    "# Entrenamiento del Random Forest\n",
    "# n_estimators=200 árboles, class_weight='balanced' para manejar desbalance de clases\n",
    "rf = RandomForestClassifier(\n",
    "    n_estimators=200,\n",
    "    random_state=RANDOM_STATE,\n",
    "    n_jobs=-1,\n",
    "    class_weight='balanced'\n",
    ")\n",
    "\n",
    "rf.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_train_pred = rf.predict(X_train_scaled)\n",
    "y_test_pred  = rf.predict(X_test_scaled)\n",
    "\n",
    "print('Modelo Random Forest entrenado exitosamente')\n",
    "print(f'Número de árboles: {rf.n_estimators}')"
]))

# --- 7.2 Evaluación train vs test ---
nuevas_celdas_rf.append(md([
    "### 7.2 Evaluación del Modelo: Train vs. Test\n",
    "\n",
    "Comparamos el rendimiento en entrenamiento y en prueba para detectar "
    "posible **sobreajuste**: si el accuracy de train es mucho mayor que el de test, "
    "el modelo memorizó los datos en lugar de aprender patrones generalizables."
]))
nuevas_celdas_rf.append(code([
    "def evaluar_modelo(nombre, y_real, y_pred):\n",
    "    \"\"\"Calcula métricas principales para un modelo de clasificación binaria.\"\"\"\n",
    "    return pd.Series({\n",
    "        'modelo': nombre,\n",
    "        'accuracy':  accuracy_score(y_real, y_pred),\n",
    "        'precision': precision_score(y_real, y_pred, zero_division=0),\n",
    "        'recall':    recall_score(y_real, y_pred, zero_division=0),\n",
    "        'f1_score':  f1_score(y_real, y_pred, zero_division=0)\n",
    "    })\n",
    "\n",
    "resultados_rf = pd.DataFrame([\n",
    "    evaluar_modelo('Random Forest - train', y_train, y_train_pred),\n",
    "    evaluar_modelo('Random Forest - test',  y_test,  y_test_pred)\n",
    "])\n",
    "\n",
    "resultados_rf.set_index('modelo').round(3)"
]))

# --- 7.3 Reporte clasificación ---
nuevas_celdas_rf.append(md([
    "### 7.3 Reporte de Clasificación\n",
    "\n",
    "- **Precision:** de los clientes predichos como churn, ¿cuántos realmente abandonaron?\n",
    "- **Recall:** de los clientes que realmente abandonaron, ¿cuántos detectó el modelo?\n",
    "- **F1-score:** media armónica entre precision y recall."
]))
nuevas_celdas_rf.append(code([
    "print('Reporte de clasificación en el conjunto de prueba:')\n",
    "print(classification_report(y_test, y_test_pred, target_names=['Activo (0)', 'Churn (1)']))"
]))

# --- 7.4 Matriz de confusión ---
nuevas_celdas_rf.append(md([
    "### 7.4 Matriz de Confusión\n",
    "\n",
    "La matriz de confusión muestra en qué casos acierta y en cuáles se equivoca el modelo:\n",
    "- **Verdaderos negativos (TN):** clientes activos correctamente clasificados\n",
    "- **Falsos positivos (FP):** clientes activos clasificados como churn\n",
    "- **Falsos negativos (FN):** clientes churn que el modelo no detectó ⚠️ (los más costosos para el negocio)\n",
    "- **Verdaderos positivos (TP):** clientes churn correctamente detectados"
]))
nuevas_celdas_rf.append(code([
    "cm_rf = confusion_matrix(y_test, y_test_pred)\n",
    "\n",
    "disp = ConfusionMatrixDisplay(\n",
    "    confusion_matrix=cm_rf,\n",
    "    display_labels=['Activo', 'Churn']\n",
    ")\n",
    "\n",
    "disp.plot(cmap='Blues', values_format='d')\n",
    "plt.title('Matriz de Confusión - Random Forest')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# --- 7.5 Importancia de variables ---
nuevas_celdas_rf.append(md([
    "### 7.5 Importancia de Variables\n",
    "\n",
    "Random Forest permite estimar qué variables fueron más útiles para dividir los datos. "
    "La importancia se calcula a partir de la **reducción promedio de impureza** (Gini) "
    "a través de todos los árboles."
]))
nuevas_celdas_rf.append(code([
    "importancias = pd.DataFrame({\n",
    "    'variable':    features,\n",
    "    'importancia': rf.feature_importances_\n",
    "}).sort_values(by='importancia', ascending=False)\n",
    "\n",
    "print('Importancia de variables:')\n",
    "importancias.round(4)"
]))
nuevas_celdas_rf.append(code([
    "plt.figure(figsize=(7, 5))\n",
    "plt.barh(importancias['variable'], importancias['importancia'], color='forestgreen')\n",
    "plt.gca().invert_yaxis()\n",
    "plt.xlabel('Importancia')\n",
    "plt.title('Importancia de Variables en Random Forest')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# --- 7.6 Exploración interna: votación ---
nuevas_celdas_rf.append(md([
    "### 7.6 Exploración Interna: ¿Cómo Votan los Árboles?\n",
    "\n",
    "Cada árbol produce una predicción individual. El Random Forest toma la **clase más votada**. "
    "Analicemos un cliente específico del conjunto de prueba para ver cómo se distribuyen los votos."
]))
nuevas_celdas_rf.append(code([
    "print(f'Número de árboles en el bosque: {len(rf.estimators_)}')\n",
    "print(type(rf.estimators_[0]))"
]))
nuevas_celdas_rf.append(code([
    "# Analizar el cliente en el índice 5 del conjunto de prueba\n",
    "idx0 = 5\n",
    "\n",
    "new_x       = X_test.iloc[[idx0]]          # conservar formato DataFrame\n",
    "new_x_sc    = X_test_scaled[idx0:idx0+1]   # versión escalada\n",
    "real_label  = y_test.iloc[idx0]\n",
    "rf_label    = rf.predict(new_x_sc)[0]\n",
    "\n",
    "print(f'Índice analizado dentro de X_test: {idx0}')\n",
    "print(f'Etiqueta real:                     {real_label}  (0=Activo, 1=Churn)')\n",
    "print(f'Predicción del Random Forest:      {rf_label}')\n",
    "\n",
    "# Predicción de cada árbol individual\n",
    "predicciones_arboles = np.array([\n",
    "    arbol.predict(new_x_sc)[0] for arbol in rf.estimators_\n",
    "])\n",
    "\n",
    "votos_0 = int(np.sum(predicciones_arboles == 0))\n",
    "votos_1 = int(np.sum(predicciones_arboles == 1))\n",
    "\n",
    "print(f'\\nVotos para Activo (0): {votos_0}')\n",
    "print(f'Votos para Churn  (1): {votos_1}')\n",
    "print(f'Primeras 20 predicciones individuales: {predicciones_arboles[:20]}')"
]))
nuevas_celdas_rf.append(code([
    "plt.figure(figsize=(5, 4))\n",
    "plt.bar(['Activo (0)', 'Churn (1)'], [votos_0, votos_1], color=['steelblue', 'salmon'])\n",
    "plt.ylabel('Número de votos')\n",
    "plt.title(f'Votación de los árboles — cliente índice {idx0}')\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# --- 7.7 Exploración de un árbol individual ---
nuevas_celdas_rf.append(md([
    "### 7.7 Exploración de un Árbol Individual\n",
    "\n",
    "Podemos inspeccionar cualquier árbol del bosque para ver su profundidad, "
    "número de hojas y sus reglas de decisión (hasta cierta profundidad para no imprimir un árbol gigante)."
]))
nuevas_celdas_rf.append(code([
    "arbol_individual = rf.estimators_[1]\n",
    "\n",
    "print(f'Profundidad del árbol:       {arbol_individual.get_depth()}')\n",
    "print(f'Número de hojas del árbol:   {arbol_individual.get_n_leaves()}')\n",
    "print()\n",
    "print('Primeras 3 capas de reglas del árbol individual:')\n",
    "reglas = export_text(arbol_individual, feature_names=features, max_depth=3)\n",
    "print(reglas)"
]))

# --- 7.8 Estadísticas de todos los árboles ---
nuevas_celdas_rf.append(md([
    "### 7.8 Estadísticas de Todos los Árboles del Bosque\n",
    "\n",
    "No todos los árboles tienen la misma profundidad ni el mismo número de hojas. "
    "Veamos la distribución de estos valores en los 200 árboles."
]))
nuevas_celdas_rf.append(code([
    "profundidades = [arbol.get_depth()    for arbol in rf.estimators_]\n",
    "hojas         = [arbol.get_n_leaves() for arbol in rf.estimators_]\n",
    "\n",
    "resumen_arboles = pd.DataFrame({'profundidad': profundidades, 'numero_hojas': hojas})\n",
    "resumen_arboles.describe().round(2)"
]))
nuevas_celdas_rf.append(code([
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "\n",
    "axes[0].hist(profundidades, bins=15, color='steelblue')\n",
    "axes[0].set_xlabel('Profundidad')\n",
    "axes[0].set_ylabel('Frecuencia')\n",
    "axes[0].set_title('Distribución de profundidades de los árboles')\n",
    "\n",
    "axes[1].hist(hojas, bins=15, color='teal')\n",
    "axes[1].set_xlabel('Número de hojas')\n",
    "axes[1].set_ylabel('Frecuencia')\n",
    "axes[1].set_title('Distribución del número de hojas de los árboles')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

# --- 7.9 Comparación RF vs árboles individuales ---
nuevas_celdas_rf.append(md([
    "### 7.9 Comparación: Random Forest vs. Árboles Individuales\n",
    "\n",
    "Comparamos tres modelos:\n",
    "1. **Random Forest completo** (200 árboles)\n",
    "2. **Mejor árbol individual dentro del bosque** (el que tiene mejor accuracy en test)\n",
    "3. **Árbol de decisión independiente** optimizado con `GridSearchCV`"
]))
nuevas_celdas_rf.append(code([
    "# Accuracy de cada árbol individual del bosque en el conjunto de prueba\n",
    "accuracies_arboles = []\n",
    "\n",
    "for arbol in rf.estimators_:\n",
    "    pred_i = arbol.predict(X_test_scaled)\n",
    "    acc_i  = accuracy_score(y_test, pred_i)\n",
    "    accuracies_arboles.append(acc_i)\n",
    "\n",
    "mejor_idx        = int(np.argmax(accuracies_arboles))\n",
    "mejor_arbol_bosque = rf.estimators_[mejor_idx]\n",
    "\n",
    "print(f'Mejor árbol dentro del bosque: índice {mejor_idx}')\n",
    "print(f'Accuracy del mejor árbol individual: {accuracies_arboles[mejor_idx]:.3f}')\n",
    "print(f'Profundidad: {mejor_arbol_bosque.get_depth()}')\n",
    "print(f'Número de hojas: {mejor_arbol_bosque.get_n_leaves()}')"
]))
nuevas_celdas_rf.append(code([
    "plt.figure(figsize=(7, 4))\n",
    "plt.hist(accuracies_arboles, bins=15, color='steelblue', alpha=0.7)\n",
    "plt.axvline(accuracy_score(y_test, y_test_pred), linestyle='--', color='red',\n",
    "            label=f'Random Forest completo ({accuracy_score(y_test, y_test_pred):.3f})')\n",
    "plt.xlabel('Accuracy en test')\n",
    "plt.ylabel('Número de árboles')\n",
    "plt.title('Accuracy de los árboles individuales vs. el bosque completo')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))
nuevas_celdas_rf.append(md([
    "#### Árbol de Decisión Independiente con GridSearchCV\n",
    "\n",
    "Entrenamos un árbol de decisión independiente y usamos **búsqueda en cuadrícula** "
    "para encontrar la mejor combinación de hiperparámetros (`max_depth`, `min_samples_leaf`)."
]))
nuevas_celdas_rf.append(code([
    "param_grid = {\n",
    "    'max_depth':        [2, 3, 4, 5, 6, None],\n",
    "    'min_samples_leaf': [1, 5, 10, 20]\n",
    "}\n",
    "\n",
    "dt_base = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced')\n",
    "\n",
    "grid_dt = GridSearchCV(\n",
    "    estimator=dt_base,\n",
    "    param_grid=param_grid,\n",
    "    scoring='f1',\n",
    "    cv=5,\n",
    "    n_jobs=-1\n",
    ")\n",
    "\n",
    "grid_dt.fit(X_train_scaled, y_train)\n",
    "\n",
    "mejor_dt         = grid_dt.best_estimator_\n",
    "y_test_pred_dt   = mejor_dt.predict(X_test_scaled)\n",
    "\n",
    "print('Mejores hiperparámetros del árbol independiente:')\n",
    "print(grid_dt.best_params_)\n",
    "print(f'Profundidad: {mejor_dt.get_depth()}')\n",
    "print(f'Número de hojas: {mejor_dt.get_n_leaves()}')"
]))
nuevas_celdas_rf.append(code([
    "# Tabla comparativa de los tres modelos\n",
    "pred_mejor_arbol_bosque = mejor_arbol_bosque.predict(X_test_scaled)\n",
    "\n",
    "comparacion_arboles = pd.DataFrame([\n",
    "    evaluar_modelo('Random Forest completo (200 árboles)', y_test, y_test_pred),\n",
    "    evaluar_modelo('Mejor árbol individual del bosque',    y_test, pred_mejor_arbol_bosque),\n",
    "    evaluar_modelo('Árbol de decisión independiente',      y_test, y_test_pred_dt)\n",
    "])\n",
    "\n",
    "print('=== COMPARACIÓN INTERNA DE ÁRBOLES ===')\n",
    "comparacion_arboles.set_index('modelo').round(3)"
]))

# --- 7.10 Experimentación con hiperparámetros ---
nuevas_celdas_rf.append(md([
    "### 7.10 Experimentación con Hiperparámetros\n",
    "\n",
    "Modificamos los hiperparámetros del Random Forest para observar cómo cambian las métricas. "
    "Basado en la actividad de la práctica de clase:\n",
    "- `n_estimators`: número de árboles\n",
    "- `max_depth`: profundidad máxima de cada árbol\n",
    "- `min_samples_leaf`: mínimo de muestras en hojas (regularización)\n",
    "- `class_weight='balanced'`: compensa el desbalance de clases"
]))
nuevas_celdas_rf.append(code([
    "rf_experimento = RandomForestClassifier(\n",
    "    n_estimators=100,\n",
    "    max_depth=5,\n",
    "    min_samples_leaf=5,\n",
    "    random_state=RANDOM_STATE,\n",
    "    n_jobs=-1,\n",
    "    class_weight='balanced'\n",
    ")\n",
    "\n",
    "rf_experimento.fit(X_train_scaled, y_train)\n",
    "\n",
    "pred_train_exp = rf_experimento.predict(X_train_scaled)\n",
    "pred_test_exp  = rf_experimento.predict(X_test_scaled)\n",
    "\n",
    "pd.DataFrame([\n",
    "    evaluar_modelo('RF Experimento - train', y_train, pred_train_exp),\n",
    "    evaluar_modelo('RF Experimento - test',  y_test,  pred_test_exp)\n",
    "]).set_index('modelo').round(3)"
]))

# ── Actualizar celda de comparación final (celdas_despues_rf[1]) ──────────────
# Esa celda usa y_pred_rf y y_probs_rf — los mantenemos pero actualizamos
# el nombre de la variable a y_test_pred para ser consistentes.
# Patch: reescribir celda de comparación final para usar y_test_pred del RF
celdas_despues_rf[1]["source"] = [
    "# Curvas ROC comparativas: Regresión Logística vs. Random Forest\n",
    "auc_rf  = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])\n",
    "fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test_scaled)[:, 1])\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7, 5))\n",
    "ax.plot(fpr_lr, tpr_lr, color='red',   label=f'Reg. Logística (AUC = {auc_lr:.2f})')\n",
    "ax.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest  (AUC = {auc_rf:.2f})')\n",
    "ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Aleatorio')\n",
    "ax.set_xlabel('Tasa de Falsos Positivos')\n",
    "ax.set_ylabel('Tasa de Verdaderos Positivos')\n",
    "ax.set_title('Comparación de Curvas ROC')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
]

celdas_despues_rf[2]["source"] = [
    "# Tabla comparativa final entre técnicas\n",
    "acc_rf  = accuracy_score(y_test, y_test_pred)\n",
    "\n",
    "comparacion_final = pd.DataFrame({\n",
    "    'Modelo':   ['Regresión Logística', 'Random Forest'],\n",
    "    'Accuracy': [acc_lr, acc_rf],\n",
    "    'ROC-AUC':  [auc_lr, auc_rf]\n",
    "})\n",
    "print('=== COMPARACIÓN FINAL DE MODELOS ===')\n",
    "print(comparacion_final.to_string(index=False))\n",
    "\n",
    "mejor = 'Regresión Logística' if auc_lr > auc_rf else 'Random Forest'\n",
    "print(f'\\nMejor modelo según ROC-AUC: {mejor}')"
]

# ── Ensamblar notebook final ───────────────────────────────────────────────────
nb["cells"] = celdas_antes_rf + nuevas_celdas_rf + celdas_despues_rf

with open("Proyecto_Churn_Streaming.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

total = len(nb["cells"])
rf_count = len(nuevas_celdas_rf)
print(f"Notebook actualizado: {total} celdas totales")
print(f"  - Sección RF reemplazada: {rf_count} celdas nuevas (metodología de la práctica)")
print(f"  - Resto del notebook conservado intacto")
