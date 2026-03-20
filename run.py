"""
Entry point for Cleaning_Analyzing_Employee_Exit_Surveys. Loads and summarizes exit surveys.
"""
import os
import pandas as pd

def main():
    base = os.path.dirname(__file__)
    dete = os.path.join(base, "dete_survey.csv")
    tafe = os.path.join(base, "tafe_survey.csv")
    if not os.path.isfile(dete):
        print("=== Cleaning_Analyzing_Employee_Exit_Surveys ===\nRun from project root. Run notebook for full analysis.")
        return
    df_d = pd.read_csv(dete)
    print("=== Cleaning_Analyzing_Employee_Exit_Surveys ===\nDETE rows:", len(df_d))
    if os.path.isfile(tafe):
        df_t = pd.read_csv(tafe)
        print("TAFE rows:", len(df_t))
    print("Full analysis: run 'Analyzing_Employee_Exit_Surveys.ipynb'.")

if __name__ == "__main__":
    main()
