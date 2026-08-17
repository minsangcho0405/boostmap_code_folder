# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# 최종 모델에 실제 채택된 6개 피처
FINAL_FEATURES = [
    "매출_저점대비_반등폭",
    "매출_모멘텀",
    "분기별_총_유동인구_수",
    "구매전환율_100cap(%)",
    "저녁심야_매출_비중(%)",
    "2030대_소비_비중",
]

# t검정 탐색 대상 후보 피처 전체(18개)
ALL_CANDIDATE_FEATURES = {
    "매출_YoY_윈저화(%)": "매출 YoY(윈저화)",
    "매출_모멘텀": "매출 모멘텀",
    "매출_저점대비_반등폭": "매출 저점대비 반등폭",
    "2030대_소비_비중": "2030대 소비 비중",
    "2030대_소비_비중_증가(%p, YoY)": "2030대 소비 비중 증가폭",
    "2030비중_추세기울기": "2030비중 추세기울기",
    "주말_매출_비중(%)": "주말 매출 비중",
    "저녁심야_매출_비중(%)": "저녁~심야 매출 비중",
    "트렌디업종_매출_비중(%)": "트렌디 업종 매출 비중",
    "유동인구_YoY증가율(%)": "유동인구 YoY 증가율",
    "분기별_총_유동인구_수": "분기별 총 유동인구 수",
    "유동인구_연속개선_분기수": "유동인구 연속개선 분기수",
    "주말비중_변화폭_2분기": "주말비중 변화폭(2분기)",
    "유동인구_YoY(%)": "유동인구 YoY(신규)",
    "전환율_%p변화": "전환율 %p변화",
    "구매전환율_100cap(%)": "구매전환율 수준",
    "조건_동시충족_직전분기": "조건 동시충족 직전분기(t-1)",
    "조건_동시충족_최근4분기_충족횟수": "조건 동시충족 최근4분기 횟수",
}


# 두 집단 간 효과크기(Cohen's d) 계산
def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan


# 마스터 학습데이터 로드 및 컬럼명 정리
def load_historical_data():
    target_file = "데이터_최종_활성화라벨_CAGR규모필터_4.xlsx"
    if not os.path.exists(target_file):
        files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
        target_file = [f for f in files if '활성화라벨' in f][0]

    master = pd.read_excel(target_file, sheet_name="최종_모델링데이터")
    master = master.rename(columns={
        "활성화_현재상태(CAGR+규모필터)_0또는1": "활성화_현재상태",
        "활성화_라벨_1년후(CAGR+규모필터)_0또는1": "활성화_라벨_1년후",
        "매출_YoY증가율_윈저화(%)": "매출_YoY_윈저화(%)",
    })
    return master


# 후보 피처별 t검정(활성화 vs 비활성화 그룹 비교, Bonferroni 보정)
def run_ttest(train_df, label_col="활성화_라벨_1년후"):
    results = []
    n_features = len(ALL_CANDIDATE_FEATURES)
    for col, label_kr in ALL_CANDIDATE_FEATURES.items():
        if col not in train_df.columns:
            continue
        sub = train_df[[col, label_col]].dropna()
        g1 = sub.loc[sub[label_col] == 1, col]
        g0 = sub.loc[sub[label_col] == 0, col]
        if len(g1) < 5 or len(g0) < 5:
            continue
        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
        p_bonf = min(p_val * n_features, 1.0)
        d = cohens_d(g1, g0)
        results.append({
            "피처": label_kr, "n(0→1)": len(g1), "n(0→0)": len(g0),
            "평균(0→1)": round(g1.mean(), 3), "평균(0→0)": round(g0.mean(), 3),
            "p-value": p_val, "p-value(Bonferroni)": p_bonf,
            "유의": "Y" if p_bonf < 0.05 else "N", "Cohen's d": round(d, 3),
        })
    return pd.DataFrame(results)


# train으로 로지스틱회귀 학습, test로 AUC/F1 평가
def train_and_evaluate(df, label_col="활성화_라벨_1년후"):
    train = df[(df["data_split"] == "train") & (df["활성화_현재상태"] == 0)]
    val = df[(df["data_split"] == "validation") & (df["활성화_현재상태"] == 0)]
    test = df[(df["data_split"] == "test") & (df["활성화_현재상태"] == 0)]

    def fit_eval(feats, tr, ev):
        tr_c = tr[feats + [label_col]].dropna()
        ev_c = ev[feats + [label_col]].dropna()
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr_c[feats])
        Xev = sc.transform(ev_c[feats])
        m = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, tr_c[label_col])
        pred = m.predict_proba(Xev)[:, 1]
        auc = roc_auc_score(ev_c[label_col], pred)
        f1 = f1_score(ev_c[label_col], (pred >= 0.5).astype(int))
        return m, sc, auc, f1

    model_official, sc_official, auc_test_official, f1_test_official = fit_eval(FINAL_FEATURES, train, test)
    return model_official, sc_official, auc_test_official, f1_test_official


# 현재 비활성화 상권(apply 대상) 예측확률 산출 및 순위화
def predict_apply_target(master_df, model, scaler):
    candidates = master_df[master_df["data_split"] == "apply_최종예측대상"].copy()
    candidates = candidates.dropna(subset=FINAL_FEATURES)

    X_new = scaler.transform(candidates[FINAL_FEATURES])
    candidates["예측확률"] = model.predict_proba(X_new)[:, 1]

    ranking = candidates.sort_values("예측확률", ascending=False).reset_index(drop=True)
    ranking["순위"] = ranking.index + 1
    return ranking[["순위", "상권_코드", "상권_코드_명", "예측확률"] + FINAL_FEATURES]


if __name__ == "__main__":
    # 1) 데이터 로드
    print("1) 학습용 이력 마스터 데이터 로드...")
    full = load_historical_data()

    # 2) t검정 + 모델 학습/평가
    print("2) 통계 검증(t-검정) 및 모델 학습...")
    train_for_ttest = full[(full["data_split"] == "train") & (full["활성화_현재상태"] == 0)]
    ttest_result = run_ttest(train_for_ttest)
    ttest_result.to_excel("t검정_결과.xlsx", index=False)

    print("\n=== t-검정 결과 (Bonferroni 보정 기준 유의한 피처) ===")
    print(ttest_result[ttest_result["유의"] == "Y"].sort_values("p-value(Bonferroni)").to_string(index=False))
    print("\n=== t-검정 결과 전체 ===")
    print(ttest_result.sort_values("p-value(Bonferroni)").to_string(index=False))

    model, scaler, auc_test, f1_test = train_and_evaluate(full)
    print(f"\n=== 최종 모델(FINAL_FEATURES) test셋 성능 ===")
    print(f"AUC: {auc_test:.4f}")
    print(f"F1 : {f1_test:.4f}")

    # 3) 예측 및 저장
    print("3) apply_최종예측대상 예측 순위 산출 중...")
    ranking_df = predict_apply_target(full, model, scaler)

    ranking_df.to_excel("예측_순위표.xlsx", index=False)
    print("\n=== 활성화 가능성 예측 Top 5 상권 ===")
    print(ranking_df[["순위", "상권_코드_명", "예측확률"]].head().to_string(index=False))

    print("\n'예측_순위표.xlsx' 저장 완료!")
