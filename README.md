# HR Analytics: Employee Attrition & Dissatisfaction (Project Elevate)

This repository transforms a basic data cleaning exercise into a **sophisticated HR analytics platform**, analyzing over 1,500 employee exit surveys from the Department of Education, Training and Employment (DETE) and the Technical and Further Education (TAFE) institute in Queensland, Australia.

By moving beyond simple aggregations, this project harmonizes disparate survey schemas, engineers a unified dissatisfaction target, maps attrition risk across tenure and demographics, and establishes a baseline Random Forest model for churn prediction.

## Project Structure

* `exit_pipeline.py` — The core reproducible HR analytics pipeline. Harmonizes data, engineers features, trains the churn model, and generates 11 static charts plus an interactive dashboard.
* `docs/report.md` — A comprehensive paper-style report detailing the methodology, attrition drivers, and strategic HR recommendations.
* `docs/dashboard.html` — An **interactive Plotly executive dashboard** exploring the attrition data.
* `docs/assets/` — 11 generated static charts supporting the report.
* `Analyzing_Employee_Exit_Surveys.ipynb` — The original legacy EDA notebook.

## Key Findings

1. **The "Seven-Year Itch":** Dissatisfaction is lowest among new hires (<1 year), but rises steadily to peak dramatically in the 7-10 year tenure bucket before declining among long-term loyalists.
2. **Institute Disparity:** DETE employees report a significantly higher rate of dissatisfaction (47.8%) compared to TAFE employees (26.6%).
3. **Management Friction:** TAFE Likert scale data reveals that the primary drivers of dissatisfaction are universally related to management support and internal communication, rather than peer relationships.
4. **Predictive Churn:** The Random Forest model demonstrates that Separation Type, Age, and Tenure are the strongest predictors of dissatisfaction-driven churn (ROC-AUC: 0.747).

Read the full analysis in [docs/report.md](docs/report.md) or open `docs/dashboard.html` in your browser to explore the data interactively.

## How to Run the Pipeline

Install the dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```

Run the pipeline:
```bash
python exit_pipeline.py
```
This will process the data, train the model, and regenerate all charts and the dashboard in the `exit_outputs/` directory.
