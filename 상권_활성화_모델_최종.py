# -*- coding: utf-8 -*-
# [최종] train+test(8~17) 전체로 재학습 후 apply(18~21) 예측 - 배포용
# LightGBM + (dcor 유의성검정 -> 상관계수 중복제거
#         -> AUC 최대화 피처조합 탐색)으로 자동 결정된 FINAL_FEATURES 채택.

import os
import re
import itertools
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

CURRENT_COL = "활성화_현재상태"
LABEL_COL = "활성화_라벨_1년후"

TRAIN_TEST_Q = range(8, 18)
APPLY_Q = range(18, 22)

# 후보 피처 18개: 컬럼명 -> (계열 구분, 표시명, 계산 개요) - dcor 유의성 검정 대상
ALL_CANDIDATE_FEATURES = {
    "매출_YoY_윈저화(%)": ("매출 계열", "매출 YoY(윈저화)", "전년동기 대비 증가율"),
    "매출_모멘텀": ("매출 계열", "매출 모멘텀", "최근2분기 평균YoY - 이전2분기 평균YoY"),
    "매출_저점대비_반등폭": ("매출 계열", "매출 저점대비 반등폭", "현재YoY - 최근3분기 중 최저YoY"),
    "2030대_소비_비중": ("매출 계열", "2030대 소비 비중", "2030세대 매출/전체"),
    "2030대_소비_비중_증가(%p, YoY)": ("매출 계열", "2030대 소비 비중 증가폭", "위 비중의 전년 대비 변화"),
    "2030비중_추세기울기": ("매출 계열", "2030비중 추세기울기", "최근4분기 OLS 회귀 기울기"),
    "주말_매출_비중(%)": ("매출 계열", "주말 매출 비중", "주말 매출/전체"),
    "저녁심야_매출_비중(%)": ("매출 계열", "저녁~심야 매출 비중", "17~24시 매출/전체"),
    "트렌디업종_매출_비중(%)": ("매출 계열", "트렌디 업종 매출 비중", "카페·호프·제과 매출/전체"),
    "유동인구_YoY증가율(%)": ("매출 계열", "유동인구 YoY 증가율", "유동인구 전년 대비"),
    "분기별_총_유동인구_수": ("매출 계열", "분기별 총 유동인구 수", "원본 그대로"),
    "유동인구_연속개선_분기수": ("매출 계열", "유동인구 연속개선 분기수", "최근3분기 YoY 개선 분기수"),
    "주말비중_변화폭_2분기": ("매출 계열", "주말비중 변화폭(2분기)", "주말비중의 2분기 전 대비 변화"),
    "유동인구_YoY(%)": ("유입/전환 계열", "유동인구 YoY(신규)", "총 유동인구 전년동분기 대비 증가율"),
    "전환율_%p변화": ("유입/전환 계열", "전환율 %p변화", "구매전환율(t) - 구매전환율(t-4)"),
    "구매전환율_100cap(%)": ("유입/전환 계열", "구매전환율 수준", "min(매출건수/유동인구×100, 100)"),
    "조건_동시충족_직전분기": ("유입/전환 계열", "조건 동시충족 직전분기(t-1)", "t-1 시점 유동인구YoY>0 & 전환율%p>0"),
    "조건_동시충족_최근4분기_충족횟수": ("유입/전환 계열", "조건 동시충족 최근4분기 횟수", "최근4분기 중 위 조건 충족횟수(0~4)"),
}

# LightGBM 하이퍼파라미터 (train 내부 5-fold CV로 사전 튜닝된 값)
MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=-1,
    learning_rate=0.1,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    verbosity=-1,
)

# 상관계수(|r|) 이 값 이상이면 "사실상 같은 정보"로 간주해, 그 군집에서 설명력
# (Distance Correlation) 1위만 남기고 나머지는 최종 피처 후보에서 제외한다.
CORR_THRESHOLD = 0.5

# 피처 조합 탐색 시 전수탐색(2^k-1개 조합) 대신 그리디 전진선택으로 전환하는 기준.
MAX_EXHAUSTIVE_COMBOS = 4096


# ============================================================
# [LightGBM 특수문자 이슈 대응] 피처명에 쉼표/괄호 등 JSON 특수문자가 있으면
# LightGBM이 에러를 내므로, 학습/예측 직전에만 안전한 이름으로 바꿔서 사용한다.
# ============================================================
def _safe_lgbm_columns(df):
    rename_map = {c: re.sub(r"[^0-9a-zA-Z가-힣_]", "_", c) for c in df.columns}
    return df.rename(columns=rename_map)


# ============================================================
# Distance Correlation + Permutation Test (속도 최적화 버전)
# y가 이진(0/1) 라벨이라는 성질을 이용해 permutation마다 N×N 행렬을 새로 만들지
# 않고 행렬-벡터 연산(BLAS)으로 배치 처리. 결과는 수학적으로 동일, 속도만 개선.
# ============================================================
def _center_matrix(m):
    return m - m.mean(axis=0) - m.mean(axis=1)[:, None] + m.mean()


def _binary_dist_matrix(y):
    y = np.asarray(y, dtype=float)
    return np.abs(y[:, None] - y[None, :])


def permutation_test(x, y, n_permutations=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float)
    N = len(y_arr)

    A_raw = squareform(pdist(x_arr))
    row_sums_a = A_raw.sum(axis=1)
    S_a_total = row_sums_a.sum()
    dvarx2 = (_center_matrix(A_raw) ** 2).mean()

    n1 = int(round(y_arr.sum()))
    n0 = N - n1
    S_b_total = 2 * n0 * n1
    B_obs = _center_matrix(_binary_dist_matrix(y_arr))
    dvary2 = (B_obs ** 2).mean()

    if dvarx2 <= 0 or dvary2 <= 0:
        return 0.0, 1.0

    def dcor_of(idx1):
        v = np.zeros(N)
        v[idx1] = 1.0
        Av = A_raw.dot(v)
        S_G1 = row_sums_a[idx1].sum()
        cross = S_G1 - v.dot(Av)
        S_ab = 2 * cross
        term5 = n1 * S_a_total + (n0 - n1) * S_G1
        numerator = S_ab - (2.0 / N) * term5 + (1.0 / N ** 2) * S_a_total * S_b_total
        dcov2 = numerator / (N ** 2)
        return float(np.sqrt(max(dcov2, 0) / np.sqrt(dvarx2 * dvary2)))

    observed = dcor_of(np.where(y_arr == 1)[0])

    V = np.zeros((N, n_permutations))
    for k in range(n_permutations):
        idx1 = rng.permutation(N)[:n1]
        V[idx1, k] = 1.0

    AV = A_raw.dot(V)
    cross_terms = (V * AV).sum(axis=0)
    S_G1_all = (row_sums_a[:, None] * V).sum(axis=0)
    S_ab_all = 2 * (S_G1_all - cross_terms)
    term5_all = n1 * S_a_total + (n0 - n1) * S_G1_all
    numerator_all = S_ab_all - (2.0 / N) * term5_all + (1.0 / N ** 2) * S_a_total * S_b_total
    dcov2_all = numerator_all / (N ** 2)
    dcor_all = np.sqrt(np.clip(dcov2_all, 0, None) / np.sqrt(dvarx2 * dvary2))

    count = int((dcor_all >= observed).sum())
    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


def run_dcor_test(df, label_col=LABEL_COL, n_permutations=1000, verbose=True):
    results = []
    total = len(ALL_CANDIDATE_FEATURES)
    for i, (col, (구분, label_kr, 설명)) in enumerate(ALL_CANDIDATE_FEATURES.items(), 1):
        if col not in df.columns:
            continue
        sub = df[[col, label_col]].dropna()
        if len(sub) < 5:
            continue
        if verbose:
            print(f"  [{i}/{total}] {label_kr}: 검정 중... (N={len(sub)})")
        dcor, p_val = permutation_test(sub[col].values, sub[label_col].values, n_permutations=n_permutations)
        results.append({
            "컬럼명": col, "피처명": label_kr, "구분": 구분, "계산 개요": 설명,
            "Distance Correlation": round(dcor, 4),
            "Permutation p-value": round(p_val, 3),
            "유의성 여부 (p<0.05)": "유의함 (O)" if p_val < 0.05 else "유의하지 않음 (X)",
        })
    return pd.DataFrame(results)


# ============================================================
# 상관계수 기반 중복 피처 제거: |r| >= CORR_THRESHOLD 인 피처 군집에서
# 설명력(Distance Correlation)이 가장 높은 1개만 남긴다.
# ============================================================
def remove_redundant_correlated_features(df, candidate_features, importance_scores,
                                          corr_threshold=CORR_THRESHOLD, verbose=True):
    feats = [c for c in candidate_features if c in df.columns]
    if len(feats) <= 1:
        return feats

    corr_matrix = df[feats].corr().abs()

    adjacency = {f: set() for f in feats}
    for i, fi in enumerate(feats):
        for fj in feats[i + 1:]:
            c = corr_matrix.loc[fi, fj]
            if pd.notna(c) and c >= corr_threshold:
                adjacency[fi].add(fj)
                adjacency[fj].add(fi)

    visited = set()
    clusters = []
    for f in feats:
        if f in visited:
            continue
        stack, comp = [f], []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            stack.extend(adjacency[cur] - visited)
        clusters.append(comp)

    kept, dropped_log = [], []
    for comp in clusters:
        if len(comp) == 1:
            kept.append(comp[0])
            continue
        comp_sorted = sorted(comp, key=lambda f: importance_scores.get(f, -1.0), reverse=True)
        best = comp_sorted[0]
        drop = comp_sorted[1:]
        kept.append(best)
        dropped_log.append((best, drop))

    if verbose:
        print(f"   -> 상관계수(|r|>={corr_threshold}) 기준 중복 피처 군집 제거 (군집당 설명력 1위만 유지)")
        if dropped_log:
            for best, drop in dropped_log:
                print(f"      유지: {best}  /  제외(중복성 높음, 설명력 낮음): {drop}")
        else:
            print("      -> 중복으로 제외된 피처 없음")

    kept_set = set(kept)
    return [f for f in feats if f in kept_set]


# ============================================================
# AUC 최대화 피처 조합 탐색 (내부 5-fold 교차검증만 사용)
# ============================================================
def _cv_auc(df, feats, label_col, cv_folds=5, model_params=MODEL_PARAMS):
    X = _safe_lgbm_columns(df[feats])
    y = df[label_col].values
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**model_params)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        pred = m.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y[va_idx], pred))
    return float(np.mean(aucs)), float(np.std(aucs))


def find_best_feature_combination(df, candidate_features, label_col,
                                   cv_folds=5, model_params=MODEL_PARAMS, verbose=True):
    candidate_features = [c for c in candidate_features if c in df.columns]
    k = len(candidate_features)
    n_combos = 2 ** k - 1
    results = []

    if n_combos <= MAX_EXHAUSTIVE_COMBOS:
        if verbose:
            print(f"   -> 전수탐색 모드: 후보 {k}개, 총 {n_combos}개 조합 (5-fold CV 기준)")
        done = 0
        for r in range(1, k + 1):
            for combo in itertools.combinations(candidate_features, r):
                auc, std = _cv_auc(df, list(combo), label_col, cv_folds, model_params)
                results.append({"피처조합": combo, "피처수": len(combo), "CV_AUC평균": auc, "CV_AUC표준편차": std})
                done += 1
                if verbose and done % 100 == 0:
                    print(f"      진행: {done}/{n_combos}  (현재까지 최고 AUC={max(x['CV_AUC평균'] for x in results):.4f})")
    else:
        if verbose:
            print(f"   -> 후보 {k}개(전수탐색 {n_combos}개는 비현실적) -> 그리디 전진선택 모드로 전환")
        selected, remaining, best_auc_so_far = [], list(candidate_features), -1.0
        while remaining:
            round_results = []
            for feat in remaining:
                combo = selected + [feat]
                auc, std = _cv_auc(df, combo, label_col, cv_folds, model_params)
                round_results.append((feat, auc, std, tuple(combo)))
            round_results.sort(key=lambda x: -x[1])
            best_feat, best_auc, best_std, best_combo = round_results[0]
            if verbose:
                print(f"      [{len(selected)+1}번째 추가] {best_feat} 추가 -> CV AUC={best_auc:.4f}")
            for feat, auc, std, combo in round_results:
                results.append({"피처조합": combo, "피처수": len(combo), "CV_AUC평균": auc, "CV_AUC표준편차": std})
            if best_auc <= best_auc_so_far:
                break
            best_auc_so_far = best_auc
            selected.append(best_feat)
            remaining.remove(best_feat)

    results_df = pd.DataFrame(results).sort_values("CV_AUC평균", ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_features = list(best_row["피처조합"])
    best_cv_auc = float(best_row["CV_AUC평균"])
    if verbose:
        print(f"   -> 최적 조합 발견: {best_features}  (CV AUC={best_cv_auc:.4f})")
    return best_features, best_cv_auc


# ============================================================
# 1) 데이터 로드
# ============================================================
master = pd.read_excel(
    os.path.join(BASE_DIR, "데이터_최종_활성화라벨_CAGR규모필터_6.xlsx"),
    sheet_name="최종_모델링데이터",
)

master = master.rename(columns={
    "활성화_현재상태(CAGR+규모필터)_0또는1": "활성화_현재상태",
    "활성화_라벨_1년후(CAGR+규모필터)_0또는1": "활성화_라벨_1년후",
    "매출_YoY증가율_윈저화(%)": "매출_YoY_윈저화(%)",
})

all_labeled = master[
    (master["분기순번"].isin(TRAIN_TEST_Q)) & (master[CURRENT_COL] == 0)
]

# ============================================================
# 2) [개선된 모델] dcor 유의성검정 -> 상관계수 중복제거 -> AUC 최대화 피처조합 탐색
#    -> LightGBM으로 배포용 최종 모델 학습 (train+test 8~17 전체 사용)
# ============================================================
print("[모델 개선] 1) Distance Correlation + Permutation Test로 피처 유의성 검정...")
train_for_search = all_labeled.dropna(subset=[LABEL_COL])
dcor_result = run_dcor_test(train_for_search, verbose=True)
significant_cols = dcor_result.loc[dcor_result["유의성 여부 (p<0.05)"] == "유의함 (O)", "컬럼명"].tolist()
print(f"   -> 유의한 피처 {len(significant_cols)}개: {significant_cols}")
if not significant_cols:
    raise ValueError("dcor 검정에서 유의한 피처가 하나도 없습니다. 데이터를 확인하세요.")

print("\n[모델 개선] 2) 상관계수 기반 중복 피처 제거...")
importance_map = dict(zip(dcor_result["컬럼명"], dcor_result["Distance Correlation"]))
search_candidates = remove_redundant_correlated_features(train_for_search, significant_cols, importance_map)
print(f"   -> 중복 제거 후 최종 탐색 후보: {len(search_candidates)}개: {search_candidates}")

print("\n[모델 개선] 3) AUC 최대화 피처 조합 탐색 (내부 5-fold 교차검증)...")
FEATURES, best_cv_auc = find_best_feature_combination(train_for_search, search_candidates, LABEL_COL)
print(f"\n   => 채택된 FINAL_FEATURES: {FEATURES}")
print(f"   => 내부 CV 기준 AUC: {best_cv_auc:.4f}")

print("\n[모델 개선] 4) LightGBM 배포용 최종 모델 학습 (train+test 8~17 전체)...")
tr = all_labeled[FEATURES + [LABEL_COL]].dropna()
Xtr = _safe_lgbm_columns(tr[FEATURES])
model = lgb.LGBMClassifier(**MODEL_PARAMS).fit(Xtr, tr[LABEL_COL])
print(f"배포용 모델 학습 완료: {len(tr):,}건(양성 {int(tr[LABEL_COL].sum())}건)")

# ============================================================
# 3) 분기별 예측 (이하 로직은 원본과 동일)
# ============================================================
QUARTERS = {18: "2025Q2_예측(2026Q2대상)", 19: "2025Q3_예측(2026Q3대상)",
            20: "2025Q4_예측(2026Q4대상)", 21: "2026Q1_예측(2027Q1대상)"}

all_preds = []
for q, label in QUARTERS.items():
    cand = master[(master["분기순번"] == q) & (master[CURRENT_COL] == 0)].dropna(subset=FEATURES).copy()

    # [수정 위치] 후보 데이터가 없는 경우 예외 처리
    if cand.empty:
        print(f"{label}: 해당 분기의 후보 데이터가 없어 예측을 건너뜁니다.")
        continue

    X = _safe_lgbm_columns(cand[FEATURES])
    cand["예측확률"] = model.predict_proba(X)[:, 1]
    cand["분기라벨"] = label
    cand["검증여부"] = "미검증(예측만)"

    ranking = cand.sort_values("예측확률", ascending=False).reset_index(drop=True)
    ranking["순위"] = ranking.index + 1
    ranking["상위비율(%)"] = (ranking["순위"] / len(ranking) * 100).round(1)

    out = ranking[["순위", "상권_코드", "상권_코드_명", "예측확률", "상위비율(%)", "검증여부"]].copy()
    out["예측확률"] = (out["예측확률"] * 100).round(1)
    all_preds.append(out.assign(분기=label))

    print(f"{label}: 후보 {len(ranking)}건, 상위 5개 확인 - "
          f"{', '.join(ranking.head(5)['상권_코드_명'].tolist())}")

if not all_preds:
    print("예측할 수 있는 분기 데이터가 없습니다.")
else:
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

    # 존재하는 분기 수에 맞춰 조건 적용 (만약 21번 분기가 없어서 3개 분기만 수행된 경우 자동 대응)
    total_valid_quarters = combined["분기"].nunique()
    persistent = summary[(summary["등장분기수"] == total_valid_quarters) & (summary["상위20%_횟수"] >= min(3, total_valid_quarters))].sort_values(
        "평균예측확률(%)", ascending=False
    )

    print()
    print(f"=== 지속 상위 후보 ({total_valid_quarters}개 분기 모두 후보 + 지속 상위 진입): {len(persistent)}개 ===")
    print(persistent.head(20).to_string(index=False))

    save_path = os.path.join(BASE_DIR, "2026_2027년_분기별_예측_및_지속후보.xlsx")
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        for label in combined["분기"].unique():
            sub = combined[combined["분기"] == label].drop(columns=["분기", "상위20%"])
            sub.to_excel(writer, sheet_name=label[:31], index=False)
        summary.sort_values("평균예측확률(%)", ascending=False).to_excel(
            writer, sheet_name="분기별_통합_요약", index=False)
        persistent.to_excel(writer, sheet_name="지속상위후보", index=False)

    print(f"\n저장 완료: {save_path}")
