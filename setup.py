"""Setup configuration for the Olist Analysis package."""
from setuptools import setup, find_packages

setup(
    name='olist-analysis',
    version='1.0.0',
    description='Olist E-Commerce Demand Planning & Logistics Performance Analysis',
    author='Long',
    python_requires='>=3.9',
    packages=find_packages(exclude=['tests', 'experiments', 'notebooks']),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'requests>=2.28.0',
        'lxml>=4.9.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'statsmodels>=0.14.0',
        'scikit-learn>=1.2.0',
        'xgboost>=1.7.0',
    ],
    extras_require={
        'nlp': ['nltk>=3.8.0', 'wordcloud>=1.9.0', 'gensim>=4.3.0'],
        'ml': ['lightgbm>=3.3.0'],
        'dev': ['pytest>=7.0.0'],
    },
)
