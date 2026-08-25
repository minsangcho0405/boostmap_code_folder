"""
상권_활성화_전처리_결과.csv -> 상권_활성화_전처리_결과.xlsx 변환 스크립트

사용법:
    python csv_to_xlsx.py

MySQL INTO OUTFILE로 만든 CSV(UTF-8)를 읽어 xlsx로 저장합니다.
필요 패키지: pip install pandas openpyxl
"""

import pandas as pd

CSV_PATH = "상권_활성화_전처리_결과.csv"
XLSX_PATH = "상권_활성화_전처리_결과.xlsx"

df = pd.read_csv(CSV_PATH, encoding="utf-8")
df.to_excel(XLSX_PATH, index=False, sheet_name="활성화_상권_전처리")

print(f"완료: {XLSX_PATH} ({len(df)}행 x {len(df.columns)}열)")
