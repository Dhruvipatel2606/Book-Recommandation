#!/usr/bin/env python3
"""
Recreate plot images that were deleted: 
- plots/price_distribution.png
- price_boxplot.png
- price_histogram.png

This script reads shopping_dataset_clean.csv (or shopping_dataset.csv as fallback),
cleans the price column, and generates basic plots.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_dataset():
	candidates = [
		'shopping_dataset_clean.csv',
		'shopping_dataset.csv'
	]
	for path in candidates:
		if os.path.exists(path):
			try:
				df = pd.read_csv(path)
				return df, path
			except Exception:
				pass
	print('No dataset found. Expected shopping_dataset_clean.csv or shopping_dataset.csv in project root.')
	sys.exit(1)


def coerce_price(series: pd.Series) -> pd.Series:
	# Convert common currency formats to float
	clean = (
		series.astype(str)
		.str.replace(',', '', regex=False)
		.str.replace('₹', '', regex=False)
		.str.extract(r'([0-9]*\.?[0-9]+)')[0]
	)
	return pd.to_numeric(clean, errors='coerce')


def ensure_plots_dir():
	if not os.path.isdir('plots'):
		os.makedirs('plots', exist_ok=True)


def plot_price_histogram(df: pd.DataFrame):
	plt.figure(figsize=(8, 5))
	df['price'].dropna().plot(kind='hist', bins=40, color='#667eea', edgecolor='white')
	plt.title('Price Histogram')
	plt.xlabel('Price')
	plt.ylabel('Count')
	plt.tight_layout()
	plt.savefig('price_histogram.png', dpi=200)
	plt.close()


def plot_price_boxplot(df: pd.DataFrame):
	plt.figure(figsize=(7, 5))
	plt.boxplot(df['price'].dropna(), vert=True, patch_artist=True,
				boxprops=dict(facecolor='#a78bfa'))
	plt.title('Price Boxplot')
	plt.ylabel('Price')
	plt.tight_layout()
	plt.savefig('price_boxplot.png', dpi=200)
	plt.close()


def plot_price_distribution_by_company(df: pd.DataFrame):
	plt.figure(figsize=(9, 5))
	companies = df['company'].dropna().astype(str).str.title()
	prices = df['price']
	valid = (~prices.isna()) & (~companies.isna())
	ax = plt.gca()
	for company in sorted(companies[valid].unique()):
		subset = prices[(companies == company) & valid]
		if subset.empty:
			continue
		subset.plot(kind='kde', ax=ax, label=company)
	plt.title('Price Distribution by Company')
	plt.xlabel('Price')
	plt.ylabel('Density')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join('plots', 'price_distribution.png'), dpi=200)
	plt.close()


def main():
	df, used_path = load_dataset()
	print(f'Loaded dataset: {used_path} (rows={len(df)})')
	if 'price' not in df.columns:
		print("Dataset doesn't contain a 'price' column. Cannot generate price plots.")
		sys.exit(1)
	# Clean price
	df = df.copy()
	df['price'] = coerce_price(df['price'])
	# Optional company column handling
	if 'company' not in df.columns:
		df['company'] = 'Unknown'
	# Create plots
	ensure_plots_dir()
	plot_price_histogram(df)
	plot_price_boxplot(df)
	plot_price_distribution_by_company(df)
	print('Recreated images:')
	print(' - price_histogram.png')
	print(' - price_boxplot.png')
	print(' - plots/price_distribution.png')

if __name__ == '__main__':
	main()
