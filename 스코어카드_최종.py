# -*- coding: utf-8 -*-
"""
2026_2027년 분기별 예측 스코어카드 생성 파이프라인
===================================================
입력 파일:
  1. 2026_2027년_분기별_예측_및_지속후보.xlsx  ← 상권_활성화_모델(최종) 결과물
  2. 데이터_최종_활성화라벨_CAGR규모필터_5.xlsx ← Lift Chart용 실제 라벨
  3. 임대동향_지역별_임대료_2024년3분기___소규모_상가.xlsx ← 임대료

출력 파일:
  2026_2027년_분기별_예측_스코어카드.xlsx
    - 분기별 예측 시트 (스코어등급 + 임대료)
    - 분기별_통합_요약 (스코어등급)
    - 지속상위후보 (통합요약 등급 매핑)
    - Lift_Chart(test기준) — train(8~13) 학습 → test(14~17) 평가

최종 피처 4개:
  매출_YoY_윈저화(%), 매출_모멘텀, 2030대_소비_비중, 전환율_%p변화
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# ── 파일 경로 ──────────────────────────────────────────────
PRED_SRC = "2026_2027년_분기별_예측_임대료결합.xlsx"
DATA_SRC = "데이터_최종_활성화라벨_CAGR규모필터_5.xlsx"
RENT_SRC = "임대동향_지역별_임대료_2024년3분기___소규모_상가.xlsx"
OUTPUT   = "2026_2027년_분기별_예측_스코어카드.xlsx"

# ── 설정 ──────────────────────────────────────────────────
FEATURES = [
    "매출_YoY_윈저화(%)",
    "매출_모멘텀",
    "2030대_소비_비중",
    "전환율_%p변화",
]
CURRENT_COL = "활성화_현재상태"
LABEL_COL   = "활성화_라벨_1년후"
TRAIN_Q     = range(8, 14)
TEST_Q      = range(14, 18)

GRADE_LABEL = {
    "1등급": "★★★★★ 최우수",
    "2등급": "★★★★☆ 우수",
    "3등급": "★★★☆☆ 양호",
    "4등급": "★★☆☆☆ 보통",
    "5등급": "★☆☆☆☆ 관심",
}

# ── STEP 0: 임대료 매핑 ────────────────────────────────────
print("STEP 0: 임대료 매핑 로드...")
df_rent = pd.read_excel(os.path.join(BASE_DIR, RENT_SRC))
df_rent = df_rent.iloc[2:].reset_index(drop=True)
df_rent.columns = [
    "No","지역대","지역중","지역소",
    "2024Q3","2024Q4","2025Q1","2025Q2",
    "2025Q3","2025Q4","2026Q1","2026Q2"
]
df_rent = df_rent[df_rent["지역소"].notna() & (df_rent["지역소"] != "지역")].copy()
df_rent["임대료_2026Q2"] = pd.to_numeric(df_rent["2026Q2"], errors="coerce")
rent_map = dict(zip(df_rent["지역소"], df_rent["임대료_2026Q2"]))

def map_rent(df_in):
    df_in = df_in.copy()
    if "임대료_지역" in df_in.columns:
        df_in["임대료_2026Q2"] = df_in["임대료_지역"].map(rent_map).fillna("--")
    else:
        df_in["임대료_2026Q2"] = "--"
    return df_in

def add_grade(df_in, prob_col="예측확률"):
    df_in = df_in.copy()
    df_in["스코어등급"] = pd.qcut(
        df_in[prob_col], q=5,
        labels=["5등급","4등급","3등급","2등급","1등급"],
        duplicates="drop"
    )
    df_in["등급_라벨"] = df_in["스코어등급"].map(GRADE_LABEL)
    return df_in

# ── STEP 1: Lift Chart 생성 (test 세트 기준) ──────────────
print("STEP 1: Lift Chart 생성 (test 2024Q2~2025Q1)...")
master = pd.read_excel(os.path.join(BASE_DIR, DATA_SRC), sheet_name="최종_모델링데이터")
master = master.rename(columns={
    "활성화_현재상태(CAGR+규모필터)_0또는1": CURRENT_COL,
    "활성화_라벨_1년후(CAGR+규모필터)_0또는1": LABEL_COL,
    "매출_YoY증가율_윈저화(%)": "매출_YoY_윈저화(%)",
})
df_labeled = master[master[CURRENT_COL] == 0].copy()

tr = df_labeled[df_labeled["분기순번"].isin(TRAIN_Q)].dropna(subset=FEATURES + [LABEL_COL])
te = df_labeled[df_labeled["분기순번"].isin(TEST_Q)].dropna(subset=FEATURES + [LABEL_COL])
print(f"   Train: {len(tr):,}건 | Test: {len(te):,}건")

sc_eval = StandardScaler()
m_eval  = LogisticRegression(class_weight="balanced", max_iter=1000)
m_eval.fit(sc_eval.fit_transform(tr[FEATURES]), tr[LABEL_COL])

te_prob = m_eval.predict_proba(sc_eval.transform(te[FEATURES]))[:, 1]
te_pred = (te_prob >= 0.5).astype(int)
auc = roc_auc_score(te[LABEL_COL], te_prob)
f1  = f1_score(te[LABEL_COL], te_pred)
print(f"   AUC={auc:.3f}  F1={f1:.3f}")

te_result = te.copy()
te_result["예측확률"] = te_prob
te_result = add_grade(te_result)
기저율 = te_result[LABEL_COL].mean() * 100

grade_summary = te_result.groupby("스코어등급", observed=True).agg(
    상권수=(LABEL_COL, "count"),
    실제활성화수=(LABEL_COL, "sum"),
    실제활성화율=(LABEL_COL, "mean"),
    평균예측확률=("예측확률", "mean"),
).reset_index()
grade_summary["실제활성화율(%)"] = (grade_summary["실제활성화율"] * 100).round(1)
grade_summary["Lift"]            = (grade_summary["실제활성화율(%)"] / 기저율).round(2)
grade_summary["평균예측확률(%)"] = (grade_summary["평균예측확률"] * 100).round(1)
grade_summary["등급_라벨"]       = grade_summary["스코어등급"].map(GRADE_LABEL)
grade_summary = grade_summary[[
    "스코어등급","등급_라벨","상권수","실제활성화수",
    "실제활성화율(%)","Lift","평균예측확률(%)"
]]
print(f"   기저율: {기저율:.1f}%")
print(grade_summary.to_string(index=False))

# ── STEP 2: 예측 시트 로드 + 등급/임대료 추가 ─────────────
print("\nSTEP 2: 예측 시트 등급/임대료 추가...")
xl = pd.ExcelFile(os.path.join(BASE_DIR, PRED_SRC))
sheet_dfs = {}

분기_시트 = [sh for sh in xl.sheet_names if "예측" in sh and "대상" in sh]
for sh in 분기_시트:
    df = pd.read_excel(xl, sheet_name=sh)
    df = add_grade(df)
    df = map_rent(df)
    sheet_dfs[sh] = df
    print(f"   [{sh}] {len(df)}건")

# ── STEP 3: 통합요약 등급 부여 ────────────────────────────
print("\nSTEP 3: 통합요약 등급 부여...")
통합 = pd.read_excel(xl, sheet_name="분기별_통합_요약")
통합 = add_grade(통합, prob_col="평균예측확률(%)")
통합 = map_rent(통합)
sheet_dfs["분기별_통합_요약"] = 통합
print(f"   [분기별_통합_요약] {len(통합)}건")

# ── STEP 4: 지속상위후보 — 통합요약 등급 매핑 ─────────────
print("\nSTEP 4: 지속상위후보 등급 매핑...")
지속 = pd.read_excel(xl, sheet_name="지속상위후보")
지속 = 지속.merge(
    통합[["상권_코드","스코어등급","등급_라벨"]],
    on="상권_코드", how="left"
)
지속 = map_rent(지속)
sheet_dfs["지속상위후보"] = 지속
print(f"   [지속상위후보] {len(지속)}건")

sheet_dfs["Lift_Chart(test기준)"] = grade_summary

# ── STEP 5: 저장 ──────────────────────────────────────────
print("\nSTEP 5: 저장...")
sheet_order = 분기_시트 + ["분기별_통합_요약", "지속상위후보", "Lift_Chart(test기준)"]
save_path = os.path.join(BASE_DIR, OUTPUT)

with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
    for sh in sheet_order:
        if sh in sheet_dfs and not sheet_dfs[sh].empty:
            sheet_dfs[sh].to_excel(writer, sheet_name=sh[:31], index=False)

print(f"\n저장 완료 → {save_path}")
print(f"시트: {sheet_order}")
