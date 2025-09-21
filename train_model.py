#!/usr/bin/env python3
"""
Train a RandomForestClassifier to predict rating_category (High if rating>=4.0),
print Accuracy/Precision/Recall/F1, and save book_rating_model.pkl.
Also saves useful plots in plots/ directory.
"""
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

INPUT_CSV = 'shopping_dataset_clean.csv'
MODEL_PATH = 'book_rating_model.pkl'
PLOTS_DIR = 'plots'


def ensure_plots_dir():
	if not os.path.isdir(PLOTS_DIR):
		os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data():
	if not os.path.exists(INPUT_CSV):
		print(f"❌ Missing {INPUT_CSV} in project root.")
		sys.exit(1)
	df = pd.read_csv(INPUT_CSV)
	# Basic cleaning
	for col in ['title', 'author', 'company']:
		if col in df.columns:
			df[col] = df[col].astype(str).str.strip()
	# Price to float
	if 'price' in df.columns:
		df['price'] = (
			df['price'].astype(str)
			.str.replace(',', '', regex=False)
			.str.replace('₹', '', regex=False)
			.str.extract(r'([0-9]*\.?[0-9]+)')[0]
		)
		df['price'] = pd.to_numeric(df['price'], errors='coerce')
	# Rating to float
	if 'rating' in df.columns:
		ser = df['rating'].astype(str)
		ser = ser.str.replace('out of 5 stars', '', regex=False)
		ser = ser.str.replace('stars', '', regex=False)
		ser = ser.str.extract(r'([0-9]*\.?[0-9]+)')[0]
		df['rating'] = pd.to_numeric(ser, errors='coerce')
	return df


def build_features(df: pd.DataFrame):
	# Target: High (1) if rating>=4.0 else Low (0)
	if 'rating' not in df.columns:
		print('❌ rating column missing; cannot train.')
		sys.exit(1)
	mask = ~df['rating'].isna()
	df = df[mask].copy()
	df['rating_category'] = (df['rating'] >= 4.0).astype(int)
	
	# Features: price, company_encoded, optional reviews_count
	X = pd.DataFrame(index=df.index)
	if 'price' in df.columns:
		X['price'] = df['price'].fillna(df['price'].median())
	else:
		X['price'] = 0.0
	# company encoding
	comp = df.get('company', pd.Series(['unknown']*len(df), index=df.index)).str.lower().fillna('unknown')
	X['company_encoded'] = comp.map({'amazon': 0, 'flipkart': 1}).fillna(2).astype(int)
	# reviews_count optional
	if 'reviews_count' in df.columns:
		rv = pd.to_numeric(df['reviews_count'], errors='coerce').fillna(0)
		X['reviews_count'] = rv
	else:
		X['reviews_count'] = 0
	
	y = df['rating_category']
	return X, y, df


def save_plots(df: pd.DataFrame, clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray):
	ensure_plots_dir()
	# Class distribution
	plt.figure(figsize=(5,4))
	df['rating_category'].value_counts().sort_index().plot(kind='bar', color=['#dc3545','#198754'])
	plt.xticks([0,1], ['Low (<4.0)','High (>=4.0)'])
	plt.ylabel('Count')
	plt.title('Class Distribution')
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=200)
	plt.close()
	# Price histogram
	if 'price' in df.columns:
		plt.figure(figsize=(6,4))
		df['price'].dropna().plot(kind='hist', bins=40, color='#667eea', edgecolor='white')
		plt.xlabel('Price')
		plt.title('Price Histogram')
		plt.tight_layout()
		plt.savefig(os.path.join(PLOTS_DIR, 'price_histogram.png'), dpi=200)
		plt.close()
	# Confusion matrix
	plt.figure(figsize=(5,4))
	ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=['Low','High'], cmap='Blues', values_format='d')
	plt.title('Confusion Matrix')
	plt.tight_layout()
	plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=200)
	plt.close()
	# Feature importance
	try:
		importances = clf.feature_importances_
		feat_names = list(X_test.columns)
		order = np.argsort(importances)[::-1]
		plt.figure(figsize=(6,4))
		plt.bar([feat_names[i] for i in order], importances[order], color='#764ba2')
		plt.title('Feature Importance (RandomForest)')
		plt.ylabel('Importance')
		plt.tight_layout()
		plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=200)
		plt.close()
	except Exception:
		pass


def main():
	df = load_data()
	X, y, df_proc = build_features(df)
	if len(X) < 10 or y.nunique() < 2:
		print('❌ Not enough data or class variety to train.')
		sys.exit(1)
	
	# 80/20 split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
	clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	
	acc = accuracy_score(y_test, pred)
	prec = precision_score(y_test, pred, zero_division=0)
	rec = recall_score(y_test, pred, zero_division=0)
	f1 = f1_score(y_test, pred, zero_division=0)
	
	print('='*60)
	print('RandomForest - Rating High(>=4.0) vs Low metrics')
	print(f'Accuracy : {acc:.4f}')
	print(f'Precision: {prec:.4f}')
	print(f'Recall   : {rec:.4f}')
	print(f'F1-score : {f1:.4f}')
	print('-'*60)
	print(classification_report(y_test, pred, digits=4))
	print('='*60)
	
	with open(MODEL_PATH, 'wb') as f:
		pickle.dump(clf, f)
	print(f'✅ Saved model to {MODEL_PATH}')
	
	# Save plots
	save_plots(df_proc, clf, X_test, y_test, pred)
	print('📈 Saved plots to plots/: class_distribution.png, price_histogram.png, confusion_matrix.png, feature_importance.png')

if __name__ == '__main__':
	main()
