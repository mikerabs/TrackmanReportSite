# run_report_test.py
import pandas as pd
from report_generator import generate_report

# === CONFIGURATION ===
# CSV path to your game data
csv_path = "/Users/johndavis/Desktop/Portable/JD.csv"

# Exact pitcher name as it appears in the 'Pitcher' column of the CSV
pitcher_name = "Davis, Johnny"

# Manually input innings pitched
innings_pitched = 4.2

# === LOAD AND RUN ===
try:
    df = pd.read_csv(csv_path)
    report_path = generate_report(df, pitcher_name, innings_pitched)
    print(f"\n Report successfully saved to:\n{report_path}")
except FileNotFoundError as e:
    print(f"\n File not found: {e.filename}")
except KeyError as e:
    print(f"\n Missing column in data: {e}")
except Exception as e:
    print(f"\n Error generating report: {e}")