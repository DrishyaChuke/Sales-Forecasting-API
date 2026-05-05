# Sales Forecasting as a Service

Item-level and national revenue forecasting for a 10-store US retailer — served via a FastAPI backend and Streamlit front end. Achieves ~95% accuracy on 7-day revenue predictions.

## What this does

Given historical sales data, this service predicts item-level demand and 7-day national revenue. Built to demonstrate an end-to-end ML service: training, serving, and a live UI — not just a notebook.

## Architecture

```
Raw Sales Data (CSV)
    │
    ▼
Feature Engineering (Python)
    │
    ├── LightGBM model → item-level forecasts
    └── Prophet model  → national revenue trends
              │
              ▼
      FastAPI backend (/predict endpoint)
              │
              ▼
      Streamlit front end (interactive UI)
```

## Tech stack

- **Models:** LightGBM, Prophet
- **API:** FastAPI
- **UI:** Streamlit
- **Containerisation:** Docker
- **Languages:** Python

## Results

- ~95% accuracy on 7-day national revenue forecasts
- 15% reduction in simulated stockout and overstock scenarios
- Trained on data from 10 stores across 3 US states

## How to run

```bash
# Clone the repo
git clone https://github.com/DrishyaChuke/Sales-Forecasting-API.git
cd Sales-Forecasting-API

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI backend
uvicorn app.main:app --reload

# In a separate terminal, run the Streamlit UI
streamlit run app/streamlit_app.py
```

## Docker

```bash
docker build -t sales-forecasting-api .
docker run -p 8000:8000 sales-forecasting-api
```
