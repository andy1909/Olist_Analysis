# Olist Logistics Performance Analysis Pipeline 🇧🇷📦

## 1. Project Overview & Objectives

This project implements a comprehensive **End-to-End Data Pipeline** designed to analyze and optimize the logistics operations of the Olist E-Commerce platform. By simulating a real-world enterprise environment, the system integrates heterogeneous data sources to provide deep insights into **Lead Time**, **On-Time Delivery (OTD) Rates**, and **Demand Forecasting**.

The pipeline is built with a focus on modularity and automation. It handles the entire data lifecycle—from ingesting raw multi-format data (CSV, JSON, XML) and enriching it with external API data (Public Holidays), to performing rigorous data cleaning and feature engineering. The final output drives strategic decision-making through advanced time-series forecasting (Holt-Winters) and prepared datasets for interactive dashboards, ultimately helping stakeholders identify bottlenecks and optimize warehouse planning.

## 2. Technical Implementation & Workflow

The pipeline execution logic, orchestrated via `main.py`, operates as a continuous and cohesive process. The workflow commences by simulating a complex data environment, aggregating orders from CSV files, customer data from JSON structures, and product catalog details from XML formats. To enhance the analytical context, the system dynamically connects to the **Nager.Date API** to fetch Brazilian public holiday data for the years 2016-2018. This external data enrichment allows for a precise analysis of delivery delays caused by non-working days.

Once ingested, the disparate data sources undergo a transformation phase where they are merged into a unified Star Schema. The system applies strict data cleaning rules, filtering specifically for 'delivered' orders to ensure calculation accuracy and imputing missing values where necessary. During this stage, critical logistics metrics are engineered, including **Actual Lead Time**, **Delay Risk Flags**, and the **Holiday Impact Index**. Simultaneously, the pipeline performs dimensionality reduction by stripping away irrelevant attributes, ensuring the dataset is optimized for performance.

In the final analytical stage, the processed master dataset is aggregated to calculate Key Performance Indicators (KPIs) across both geographical (State-level) and temporal (Monthly trend) dimensions. The system employs the **Holt-Winters Exponential Smoothing** algorithm to generate a 12-week demand forecast, enabling proactive resource allocation. The process concludes by exporting these insights into static visualizations for immediate reporting and clean CSV datasets ready for integration with Business Intelligence tools like Power BI or Tableau.

## 3. Repository Structure

```text
Logistics_Analysis/
├── data/                        # Raw & Simulated Inputs (CSV, JSON, XML)
├── modules/                     # Core Logic Packages
│   ├── data_ingestion.py        # API handling & Format conversion
│   ├── data_processing.py       # ETL, Cleaning & Feature Engineering
│   ├── data_analytics.py        # KPI Aggregation & Forecasting models
│   └── visualization.py         # Static charting & Reporting
├── outputs/                     # Final Results (Charts, Forecasts, Master Data)
├── main.py                      # Pipeline Orchestrator
├── requirements.txt             # Project Dependencies
└── README.md                    # Documentation

## 4. Installation & Execution

To replicate this analysis, ensure you have Python 3.8+ installed. The project is designed to be plug-and-play:

    Environment Setup: Clone the repository and install the required dependencies listed in requirements.txt using pip install -r requirements.txt.

    Pipeline Execution: Run the main.py script. This single command triggers the entire sequence of ingestion, processing, modeling, and visualization.

    Result Interpretation: Upon completion, navigate to the outputs/ directory to access the cleaned Master_Logistics_Data.csv for dashboarding, or view the generated charts in outputs/Charts/ for immediate insights.
```
