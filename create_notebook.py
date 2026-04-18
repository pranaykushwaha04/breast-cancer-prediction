import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Breast Cancer Prediction\n",
    "This notebook trains a machine learning model to predict breast cancer (Malignant vs. Benign) using the provided dataset (`data.csv`).\n",
    "The goal is high accuracy and high recall."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv('data.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Preprocessing\n",
    "Cleaning up columns, handling missing values, and mapping diagnosis classes (M -> 1, B -> 0)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop unneeded columns\n",
    "if 'Unnamed: 32' in df.columns:\n",
    "    df.drop('Unnamed: 32', axis=1, inplace=True)\n",
    "if 'id' in df.columns:\n",
    "    df.drop('id', axis=1, inplace=True)\n",
    "\n",
    "# Map diagnosis: M -> 1, B -> 0\n",
    "df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})\n",
    "\n",
    "# Separate features and target\n",
    "X = df.drop('diagnosis', axis=1)\n",
    "y = df['diagnosis']\n",
    "\n",
    "print(f\"Dataset shape: {X.shape}\")\n",
    "print(f\"Class Distribution:\\n{y.value_counts(normalize=True)*100}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Splitting and Scaling Data\n",
    "We use stratified splitting to ensure the train and test sets have similar proportions of classes, and standard split for scaling."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print(\"Training records:\", len(X_train))\n",
    "print(\"Test records:\", len(X_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Training\n",
    "We choose a Random Forest Classifier. To optimize for recall (minimizing false negatives), we adjust the `class_weight` to roughly penalize misclassifying the minority class (Malignant) more heavily. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Utilizing class weights to bump up recall for class '1' (Malignant)\n",
    "rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight={0: 1, 1: 2})\n",
    "rf_model.fit(X_train_scaled, y_train)\n",
    "\n",
    "y_pred = rf_model.predict(X_test_scaled)\n",
    "y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluation Metrics\n",
    "Evaluating the model using Accuracy, Recall, Precision, F1-Score, and Confusion Matrix."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "recall = recall_score(y_test, y_pred)\n",
    "precision = precision_score(y_test, y_pred)\n",
    "f1 = f1_score(y_test, y_pred)\n",
    "\n",
    "print(f\"Accuracy : {accuracy*100:.2f}%\")\n",
    "print(f\"Recall   : {recall*100:.2f}%\")\n",
    "print(f\"Precision: {precision*100:.2f}%\")\n",
    "print(f\"F1-Score : {f1*100:.2f}%\")\n",
    "\n",
    "print(\"\\nClassification Report:\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(6,4))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])\n",
    "plt.title('Confusion Matrix')\n",
    "plt.ylabel('True Label')\n",
    "plt.xlabel('Predicted Label')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('c:/Users/Pranay/Desktop/MLproject2/train_breast_cancer.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
    print("Notebook created successfully.")
