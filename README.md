# Retail Data Platform – End-to-End Data Science Final Project

## Project Overview
This project implements an end-to-end data science and machine learning platform to predict retail
sales behavior using a modern, cloud-native data engineering stack.  
The solution follows industry best practices in data engineering, MLOps, and model deployment,
and includes an interactive Streamlit demo application.

This project is developed as an **individual final project** for a Data Science bootcamp.

---

## Business Problem
Retail organizations need accurate sales predictions to support pricing strategies,
inventory planning, and operational decision-making.  
The objective of this project is to design and implement a scalable data platform that ingests
raw retail data, transforms it into analytics-ready datasets, and trains machine learning models
to predict retail sales.

---

## Dataset
- **Source:** Kaggle – Warehouse and Retail Sales
- **Link:** [Warehouse and Retail Sales Dataset en Kaggle](https://www.kaggle.com/datasets/divyeshardeshana/warehouse-and-retail-sales)
- **Format:** CSV  
- **Domain:** Retail / Sales  
- **Raw Data Handling:**  
  Raw data is stored in an immutable data lake layer and is never modified directly, ensuring
  data lineage and reproducibility.

---

## Tech Stack (Planned)
### Data Engineering
- Python
- CSV / Amazon S3
- Data Warehouse
- dbt
- SQL

### Infrastructure & DevOps
- Docker
- Kubernetes
- Terraform
- Amazon Web Services (AWS)

### Machine Learning & MLOps
- scikit-learn / XGBoost
- Model versioning and evaluation
- Centralized logging and observability

### Application & GenAI
- Streamlit
- Generative AI (LLMs / RAG)
- Interactive demo application

---

## Project Status
- [x] Project initialization and repository setup  
- [x] Raw data ingestion pipeline with logging  
- [ ] Data cleaning and transformation  
- [ ] Exploratory Data Analysis (EDA) and Tableau dashboards  
- [ ] Machine learning model training and evaluation  
- [ ] Generative AI integration  
- [ ] Streamlit demo application  
- [ ] Final presentation  

---

## Next Steps
- Implement data cleaning and transformation layer
- Containerize the ingestion pipeline using Docker
- Load processed data into a data warehouse
- Perform EDA and feature engineering
- Train and evaluate machine learning models

