"""
데이터_최종_활성화라벨_CAGR규모필터_5.csv -> 데이터_최종_활성화라벨_CAGR규모필터_5.xlsx 변환 스크립트

사용법:
    python csv_to_xlsx.py

MySQL INTO OUTFILE로 만든 CSV(UTF-8)를 읽어 xlsx로 저장합니다.
필요 패키지: pip install pandas openpyxl
"""

import pandas as pd

CSV_PATH = "데이터_최종_활성화라벨_CAGR규모필터_5.csv"
XLSX_PATH = "데이터_최종_활성화라벨_CAGR규모필터_5.xlsx"

df = pd.read_csv(CSV_PATH, encoding="utf-8")
df.to_excel(XLSX_PATH, index=False, sheet_name="최종_모델링데이터")

print(f"완료: {XLSX_PATH} ({len(df)}행 x {len(df.columns)}열)")
