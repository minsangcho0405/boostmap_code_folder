# -*- coding: utf-8 -*-
"""
2025년 4개 분기(Q1~Q4) 피처로 2026년 각 분기를 예측하고,
여러 분기에서 공통으로 상위권에 드는 '지속 후보'를 추린다.

주의: 2025Q1→2026Q1만 실제 결과로 검증됨(AUC 0.590).
      Q2~Q4는 아직 실제 결과가 없어 예측만 가능하고 검증되지 않음.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["매출_저점대비_반등폭", "매출_모멘텀", "분기별_총_유동인구_수",
            "구매전환율_100cap(%)", "저녁심야_매출_비중(%)", "2030대_소비_비중"]
CURRENT_COL = "활성화_현재상태(CAGR+규모필터)_0또는1"
LABEL_COL = "활성화_라벨_1년후(CAGR+규모필터)_0또는1"

master = pd.read_excel("데이터_최종_활성화라벨_CAGR규모필터.xlsx", sheet_name="최종_모델링데이터")

# ---------------------------------------------------------
# 1. 배포용 최종 모델 (라벨 보유 전체: train+validation+test)
# ---------------------------------------------------------
all_labeled = master[
    (master["data_split"].isin(["train", "validation", "test"])) & (master[CURRENT_COL] == 0)
]
tr = all_labeled[FEATURES + [LABEL_COL]].dropna()
scaler = StandardScaler()
Xtr = scaler.fit_transform(tr[FEATURES])
model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, tr[LABEL_COL])
print(f"배포용 모델 학습 완료: {len(tr):,}건(양성 {int(tr[LABEL_COL].sum())}건)")

# ---------------------------------------------------------
# 2. 2025년 4개 분기 각각 예측
# ---------------------------------------------------------
QUARTERS = {17: "2025Q1_예측(2026Q1대상)", 18: "2025Q2_예측(2026Q2대상)",
            19: "2025Q3_예측(2026Q3대상)", 20: "2025Q4_예측(2026Q4대상)"}

all_preds = []
for q, label in QUARTERS.items():
    cand = master[(master["분기순번"] == q) & (master[CURRENT_COL] == 0)].dropna(subset=FEATURES).copy()
    X = scaler.transform(cand[FEATURES])
    cand["예측확률"] = model.predict_proba(X)[:, 1]
    cand["분기라벨"] = label
    cand["검증여부"] = "실제결과 확인완료" if q == 17 else "미검증(예측만)"

    ranking = cand.sort_values("예측확률", ascending=False).reset_index(drop=True)
    ranking["순위"] = ranking.index + 1
    ranking["상위비율(%)"] = (ranking["순위"] / len(ranking) * 100).round(1)

    out = ranking[["순위", "상권_코드", "상권_코드_명", "예측확률", "상위비율(%)", "검증여부"]].copy()
    out["예측확률"] = (out["예측확률"] * 100).round(1)
    all_preds.append(out.assign(분기=label))

    print(f"{label}: 후보 {len(ranking)}건, 상위 5개 확인 - "
          f"{', '.join(ranking.head(5)['상권_코드_명'].tolist())}")

# ---------------------------------------------------------
# 3. 4개 분기 통합 및 지속 상위 후보 추출
# ---------------------------------------------------------
combined = pd.concat(all_preds, ignore_index=True)

pivot_prob = combined.pivot_table(index=["상권_코드", "상권_코드_명"], columns="분기",
                                   values="예측확률", aggfunc="first")
combined["상위20%"] = combined["상위비율(%)"] <= 20
pivot_top20 = combined.pivot_table(index=["상권_코드", "상권_코드_명"], columns="분기",
                                    values="상위20%", aggfunc="first")

appear_count = pivot_prob.notna().sum(axis=1)
top20_count = pivot_top20.sum(axis=1)
avg_prob = pivot_prob.mean(axis=1)
min_prob = pivot_prob.min(axis=1)

summary = pd.DataFrame({
    "등장분기수": appear_count,
    "상위20%_횟수": top20_count,
    "평균예측확률(%)": avg_prob.round(1),
    "최저예측확률(%)": min_prob.round(1),
}).reset_index()

persistent = summary[(summary["등장분기수"] == 4) & (summary["상위20%_횟수"] >= 3)].sort_values(
    "평균예측확률(%)", ascending=False
)

print()
print(f"=== 지속 상위 후보 (4개 분기 모두 후보 + 3회 이상 상위20% 진입): {len(persistent)}개 ===")
print(persistent.head(20).to_string(index=False))

# ---------------------------------------------------------
# 4. 저장
# ---------------------------------------------------------
with pd.ExcelWriter("2026년_분기별_예측_및_지속후보.xlsx", engine="openpyxl") as writer:
    for q, label in QUARTERS.items():
        sub = combined[combined["분기"] == label].drop(columns=["분기", "상위20%"])
        sub.to_excel(writer, sheet_name=label[:31], index=False)
    summary.sort_values("평균예측확률(%)", ascending=False).to_excel(
        writer, sheet_name="분기별_통합_요약", index=False)
    persistent.to_excel(writer, sheet_name="지속상위후보", index=False)

print("\n저장 완료: 2026년_분기별_예측_및_지속후보.xlsx")
