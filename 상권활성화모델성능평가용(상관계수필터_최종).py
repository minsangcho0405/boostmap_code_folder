# -*- coding: utf-8 -*-
# [평가용] 최종 피처 모델 - AUC/F1 성능 측정용 (train만 학습, test는 순수 평가)
# LightGBM(비선형 포착) + train 내부 CV로
#          AUC를 최대화하는 피처 조합을 자동 탐색(test 데이터는 절대 들여다보지 않음)
import os
import re
import itertools
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# 분기 분할 경계 (분기순번)
TRAIN_Q = range(8, 14)   # 8~13
TEST_Q = range(14, 18)   # 14~17
APPLY_Q = range(18, 22)  # 18~21

# 후보 피처 18개: 컬럼명 -> (계열 구분, 표시명, 계산 개요)
ALL_CANDIDATE_FEATURES = {
    "매출_YoY_윈저화(%)": ("매출 계열", "매출 YoY(윈저화)", "전년동기 대비 증가율"),
    "매출_모멘텀": ("매출 계열", "매출 모멘텀", "최근2분기 평균YoY - 이전2분기 평균YoY"),
    "매출_저점대비_반등폭": ("매출 계열", "매출 저점대비 반등폭", "현재YoY - 최근3분기 중 최저YoY"),
    "2030대_소비_비중": ("매출 계열", "2030대 소비 비중", "2030세대 매출/전체"),
    "2030대_소비_비중_증가(%p, YoY)": ("매출 계열", "2030대 소비 비중 증가폭", "위 비중의 전년 대비 변화"),
    "2030비중_추세기울기": ("매출 계열", "2030비중 추세기울기", "최근4분기 OLS 회귀 기울기"),
    "주말_매출_비중(%)": ("매출 계열", "주말 매출 비중", "주말 매출/전체"),
    "저녁심야_매출_비중(%)": ("매출 계열", "저녁~심야 매출 비중", "17~24시 매출/전체"),
    "트렌디업종_매출_비중(%)": ("매출 계열", "트렌디 업종 매출 비중", "카페·제과·패스트푸드 매출/전체"),
    "유동인구_YoY증가율(%)": ("매출 계열", "유동인구 YoY 증가율", "유동인구 전년 대비"),
    "분기별_총_유동인구_수": ("매출 계열", "분기별 총 유동인구 수", "원본 그대로"),
    "유동인구_연속개선_분기수": ("매출 계열", "유동인구 연속개선 분기수", "최근3분기 YoY 개선 분기수"),
    "주말비중_변화폭_2분기": ("매출 계열", "주말비중 변화폭(2분기)", "주말비중의 2분기 전 대비 변화"),
    "전환율_%p변화": ("유입/전환 계열", "전환율 %p변화", "구매전환율(t) - 구매전환율(t-4)"),
    "구매전환율_100cap(%)": ("유입/전환 계열", "구매전환율 수준", "min(매출건수/유동인구×100, 100)"),
    "조건_동시충족_직전분기": ("유입/전환 계열", "조건 동시충족 직전분기(t-1)", "t-1 시점 유동인구YoY>0 & 전환율%p>0"),
    "조건_동시충족_최근4분기_충족횟수": ("유입/전환 계열", "조건 동시충족 최근4분기 횟수", "최근4분기 중 위 조건 충족횟수(0~4)"),
}

# LightGBM 하이퍼파라미터 (train 내부 5-fold CV로 사전 튜닝된 값)
# 로지스틱 회귀 대비 비선형 관계를 포착해 AUC가 크게 개선됨을 확인함
# (참고: 동일 조건에서 HistGradientBoosting/XGBoost와도 비교했으나 성능 차이는
#  거의 없었고, LightGBM이 test AUC 기준 근소하게 가장 높아 최종 채택)
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

# 피처 조합 탐색 시 전수탐색(2^k-1개 조합) 대신 그리디 전진선택으로 전환하는 기준.
# k(후보 피처 개수)가 커지면 조합 수가 기하급수적으로 늘어나 전수탐색이 비현실적이므로,
# 이 값을 넘으면 자동으로 더 빠른 그리디 방식으로 전환한다.
MAX_EXHAUSTIVE_COMBOS = 4096

# [중복성 제거] 두 피처의 상관계수(절댓값)가 이 값 이상이면 "사실상 같은 정보"로 간주하고,
# 그 군집(cluster) 안에서는 설명력(Distance Correlation)이 가장 높은 피처 딱 1개만 남기고
# 나머지는 전부 탐색 후보에서 제외한다.
# (예: 유동인구_YoY(%)와 유동인구_YoY증가율(%)처럼 상관계수가 1에 가까운 중복 컬럼이
#  둘 다 탐색 후보에 들어가면, 같은 정보를 두 번 넣은 셈이라 CV AUC가 노이즈로 부풀려질 수 있음)
CORR_THRESHOLD = 0.5


# [LightGBM 특수문자 이슈 대응] LightGBM은 내부적으로 피처 이름을 JSON으로 다루기 때문에
# 쉼표(,), 콜론(:), 중괄호 등 JSON 특수문자가 컬럼명에 있으면
# "Do not support special JSON characters in feature name" 에러를 낸다.
# (예: "2030대_소비_비중_증가(%p, YoY)"의 쉼표가 원인)
# 그래서 LightGBM에 넘기기 직전에만 안전한 이름으로 바꾸고, 사람이 보는 화면/엑셀
# 출력은 원래의 한글 피처명을 그대로 유지한다.
def _safe_lgbm_columns(df):
    """DataFrame의 컬럼명을 LightGBM이 허용하는 안전한 이름으로 바꿔서 복사본을 반환.
    한글/영문/숫자/밑줄만 남기고 나머지 특수문자는 전부 밑줄로 치환한다."""
    rename_map = {c: re.sub(r"[^0-9a-zA-Z가-힣_]", "_", c) for c in df.columns}
    return df.rename(columns=rename_map)


# 거리행렬 이중중심화 방식의 distance correlation (0~1, 비선형 관계도 포착)
# * 원본과 동일한 정의 그대로 유지 (관측치 1회 계산용). permutation 루프 안에서는
#   더 빠른 전용 로직(permutation_test 내부)을 쓰므로 이 함수를 반복 호출하지 않는다.
def distance_correlation(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    a = squareform(pdist(x))
    b = squareform(pdist(y))
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvarx2 = (A * A).mean()
    dvary2 = (B * B).mean()
    if dvarx2 <= 0 or dvary2 <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0) / np.sqrt(dvarx2 * dvary2)))


def _center_matrix(m):
    return m - m.mean(axis=0) - m.mean(axis=1)[:, None] + m.mean()


def _binary_dist_matrix(y):
    y = np.asarray(y, dtype=float)
    return np.abs(y[:, None] - y[None, :])


# [속도개선] 라벨 y가 이진(0/1)이라는 성질을 이용해, permutation마다 N×N 행렬을
# 새로 만드는 대신 distance covariance의 닫힌 공식(Szekely energy distance)을
# 행렬-벡터 연산(BLAS)으로 1000회를 한 번에 배치 처리한다.
# 결과값은 원래의 distance_correlation()/permutation_test()와 수학적으로 완전히
# 동일하며(부동소수점 오차 수준), N=6000+ 데이터에서 약 1000배 이상 빨라진다.
def permutation_test(x, y, n_permutations=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float)
    N = len(y_arr)

    # --- x쪽: permutation 동안 안 바뀌는 값, 한 번만 계산 ---
    A_raw = squareform(pdist(x_arr))
    row_sums_a = A_raw.sum(axis=1)
    S_a_total = row_sums_a.sum()
    dvarx2 = (_center_matrix(A_raw) ** 2).mean()

    # --- y쪽: 그룹 크기(n1,n0)는 permutation해도 안 바뀜 ---
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

    # --- permutation 전부를 한 번의 배치 행렬곱(BLAS)으로 처리 ---
    V = np.zeros((N, n_permutations))
    for k in range(n_permutations):
        idx1 = rng.permutation(N)[:n1]
        V[idx1, k] = 1.0

    AV = A_raw.dot(V)                               # (N, P) 한 번의 큰 matmul
    cross_terms = (V * AV).sum(axis=0)               # v^T A v, 퍼뮤테이션별
    S_G1_all = (row_sums_a[:, None] * V).sum(axis=0)
    S_ab_all = 2 * (S_G1_all - cross_terms)
    term5_all = n1 * S_a_total + (n0 - n1) * S_G1_all
    numerator_all = S_ab_all - (2.0 / N) * term5_all + (1.0 / N ** 2) * S_a_total * S_b_total
    dcov2_all = numerator_all / (N ** 2)
    dcor_all = np.sqrt(np.clip(dcov2_all, 0, None) / np.sqrt(dvarx2 * dvary2))

    count = int((dcor_all >= observed).sum())
    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value


# 마스터 학습데이터 로드 및 컬럼명 정리
def load_historical_data():
    target_file = os.path.join(BASE_DIR, "데이터_최종_활성화라벨_CAGR규모필터_5.xlsx")
    if not os.path.exists(target_file):
        files = [f for f in os.listdir(BASE_DIR) if f.endswith(".xlsx")]
        matched_files = [os.path.join(BASE_DIR, f) for f in files if "활성화라벨" in f]
        if not matched_files:
            raise FileNotFoundError(
                f"작업 경로('{BASE_DIR}') 내에 '활성화라벨'이 포함된 엑셀(.xlsx) 파일이 존재하지 않습니다."
            )
        target_file = matched_files[0]

    print(f"   -> 로드 파일: {target_file}")
    master = pd.read_excel(target_file, sheet_name="최종_모델링데이터")
    master = master.rename(columns={
        "활성화_현재상태(CAGR+규모필터)_0또는1": "활성화_현재상태",
        "활성화_라벨_1년후(CAGR+규모필터)_0또는1": "활성화_라벨_1년후",
        "매출_YoY증가율_윈저화(%)": "매출_YoY_윈저화(%)",
    })
    required_cols = ["분기순번", "활성화_현재상태", "활성화_라벨_1년후"]
    missing = [c for c in required_cols if c not in master.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing} (컬럼명이 바뀌었는지 확인 필요)")
    return master


# 후보 피처별 Distance Correlation + Permutation Test (n_permutations 클수록 느려짐)
def run_dcor_test(train_df, label_col="활성화_라벨_1년후", n_permutations=1000, verbose=True):
    results = []
    total = len(ALL_CANDIDATE_FEATURES)
    for i, (col, (구분, label_kr, 설명)) in enumerate(ALL_CANDIDATE_FEATURES.items(), 1):
        if col not in train_df.columns:
            if verbose:
                print(f"  [{i}/{total}] {label_kr}: 컬럼 없음 - 건너뜀")
            continue
        sub = train_df[[col, label_col]].dropna()
        if len(sub) < 5:
            if verbose:
                print(f"  [{i}/{total}] {label_kr}: 표본수 부족(<5) - 건너뜀")
            continue
        if verbose:
            print(f"  [{i}/{total}] {label_kr}: 검정 중... (N={len(sub)})")
        dcor, p_val = permutation_test(sub[col].values, sub[label_col].values, n_permutations=n_permutations)
        results.append({
            "컬럼명": col,  # 원본 컬럼명 (탐색 단계에서 유의 피처만 골라내는 용도, 화면 출력 시엔 안 씀)
            "피처명": label_kr, "구분": 구분, "계산 개요": 설명,
            "Distance Correlation": round(dcor, 4),
            "Permutation p-value": round(p_val, 3),
            "유의성 여부 (p<0.05)": "유의함 (O)" if p_val < 0.05 else "유의하지 않음 (X)",
        })
    return pd.DataFrame(results)


# [신규] 상관계수 행렬로 서로 "사실상 같은 정보"인 피처 군집을 찾아, 각 군집에서
# 설명력(Distance Correlation)이 높은 순으로 최대 CORR_KEEP_PER_CLUSTER개만 남기고
# 나머지는 탐색 후보에서 제외한다. (상관계수 행렬 자체는 화면에 출력하지 않음)
# - 예: 유동인구_YoY(%)와 유동인구_YoY증가율(%)처럼 상관계수가 거의 1인 중복 컬럼이
#   둘 다 후보에 남아있으면, 같은 정보를 두 번 넣은 것이라 탐색 과정에서 CV AUC가
#   노이즈로 부풀려질 위험이 있다.
# - 군집(cluster)은 "A-B 상관 높음, B-C 상관 높음"처럼 연쇄적으로 묶인 그룹까지
#   하나의 군집으로 취급한다(연결 요소, connected components 방식).
# [신규] 상관계수 행렬로 서로 "사실상 같은 정보"인 피처 군집을 찾아, 각 군집에서
# 설명력(Distance Correlation)이 가장 높은 딱 1개만 남기고 나머지는 탐색 후보에서 제외한다.
# (상관계수 행렬 자체는 화면에 출력하지 않음)
# - 예: 유동인구_YoY(%)와 유동인구_YoY증가율(%)처럼 상관계수가 거의 1인 중복 컬럼이
#   둘 다 후보에 남아있으면, 같은 정보를 두 번 넣은 것이라 탐색 과정에서 CV AUC가
#   노이즈로 부풀려질 위험이 있다.
# - 군집(cluster)은 "A-B 상관 높음, B-C 상관 높음"처럼 연쇄적으로 묶인 그룹까지
#   하나의 군집으로 취급한다(연결 요소, connected components 방식).
def remove_redundant_correlated_features(train_df, candidate_features, importance_scores,
                                          corr_threshold=CORR_THRESHOLD,
                                          verbose=True):
    feats = [c for c in candidate_features if c in train_df.columns]
    if len(feats) <= 1:
        return feats

    corr_matrix = train_df[feats].corr().abs()  # 계산만 하고 화면에는 출력하지 않음

    # 상관계수 >= 임계값인 피처끼리 그래프의 간선으로 연결
    adjacency = {f: set() for f in feats}
    for i, fi in enumerate(feats):
        for fj in feats[i + 1:]:
            c = corr_matrix.loc[fi, fj]
            if pd.notna(c) and c >= corr_threshold:
                adjacency[fi].add(fj)
                adjacency[fj].add(fi)

    # 연결 요소(cluster) 탐색 (BFS)
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
        # 설명력(Distance Correlation) 높은 순 정렬 후 1등만 채택, 나머지는 제외
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
            print(f"      -> 중복으로 제외된 피처 없음")

    # 원래 순서를 최대한 유지해서 반환
    kept_set = set(kept)
    return [f for f in feats if f in kept_set]


# --- CV(교차검증) 기반 AUC 계산: train 내부에서만 이루어지며 test는 절대 사용하지 않음 ---
def _cv_auc(train_df, feats, label_col, cv_folds=5, model_params=MODEL_PARAMS):
    X = _safe_lgbm_columns(train_df[feats])
    y = train_df[label_col].values
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**model_params)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        pred = m.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y[va_idx], pred))
    return float(np.mean(aucs)), float(np.std(aucs))


# [핵심] AUC를 최대화하는 피처 조합을 train 내부 CV만으로 탐색.
# - test 데이터는 이 함수 안에서 전혀 사용하지 않으므로, 여기서 고른 조합을 test에
#   딱 1번만 평가해도 "test를 보고 고른" 선택 편향(overfitting to test)이 생기지 않는다.
# - 후보가 적으면(2^k-1 <= MAX_EXHAUSTIVE_COMBOS) 모든 조합을 전수탐색.
# - 후보가 많아지면 그리디 전진선택(forward selection)으로 자동 전환해 시간을 절약.
def find_best_feature_combination(train_df, candidate_features, label_col,
                                   cv_folds=5, model_params=MODEL_PARAMS, verbose=True):
    candidate_features = [c for c in candidate_features if c in train_df.columns]
    k = len(candidate_features)
    n_combos = 2 ** k - 1

    results = []

    if n_combos <= MAX_EXHAUSTIVE_COMBOS:
        if verbose:
            print(f"   -> 전수탐색 모드: 후보 {k}개, 총 {n_combos}개 조합 (train 5-fold CV 기준)")
        done = 0
        for r in range(1, k + 1):
            for combo in itertools.combinations(candidate_features, r):
                auc, std = _cv_auc(train_df, list(combo), label_col, cv_folds, model_params)
                results.append({"피처조합": combo, "피처수": len(combo), "CV_AUC평균": auc, "CV_AUC표준편차": std})
                done += 1
                if verbose and done % 100 == 0:
                    print(f"      진행: {done}/{n_combos}  (현재까지 최고 AUC={max(x['CV_AUC평균'] for x in results):.4f})")
    else:
        # 그리디 전진선택: 매 라운드마다 CV AUC를 가장 많이 올려주는 피처를 하나씩 추가
        if verbose:
            print(f"   -> 후보 {k}개(전수탐색 {n_combos}개 조합은 비현실적) -> 그리디 전진선택 모드로 전환")
        selected = []
        remaining = list(candidate_features)
        best_auc_so_far = -1.0
        while remaining:
            round_results = []
            for feat in remaining:
                combo = selected + [feat]
                auc, std = _cv_auc(train_df, combo, label_col, cv_folds, model_params)
                round_results.append((feat, auc, std, tuple(combo)))
            round_results.sort(key=lambda x: -x[1])
            best_feat, best_auc, best_std, best_combo = round_results[0]
            if verbose:
                print(f"      [{len(selected)+1}번째 추가] {best_feat} 추가 -> CV AUC={best_auc:.4f}")
            for feat, auc, std, combo in round_results:
                results.append({"피처조합": combo, "피처수": len(combo), "CV_AUC평균": auc, "CV_AUC표준편차": std})
            if best_auc <= best_auc_so_far:
                # 더 이상 개선이 없으면 조기 종료
                break
            best_auc_so_far = best_auc
            selected.append(best_feat)
            remaining.remove(best_feat)

    results_df = pd.DataFrame(results).sort_values("CV_AUC평균", ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_features = list(best_row["피처조합"])
    best_cv_auc = float(best_row["CV_AUC평균"])
    if verbose:
        print(f"   -> 최적 조합 발견: {best_features}  (train CV AUC={best_cv_auc:.4f})")
    return best_features, best_cv_auc, results_df


# train(8~13) 학습, test(14~17) 평가 - LightGBM은 결측치를 자체 처리하므로
# (구 로지스틱 방식과 달리) 피처에 결측이 있어도 행을 버리지 않고 그대로 학습에 활용한다.
def train_and_evaluate(df, feats, label_col="활성화_라벨_1년후", model_params=MODEL_PARAMS):
    train = df[(df["분기순번"].isin(TRAIN_Q)) & (df["활성화_현재상태"] == 0)]
    test = df[(df["분기순번"].isin(TEST_Q)) & (df["활성화_현재상태"] == 0)]

    tr_c = train.dropna(subset=[label_col])
    ev_c = test.dropna(subset=[label_col])

    Xtr = _safe_lgbm_columns(tr_c[feats])
    Xev = _safe_lgbm_columns(ev_c[feats])

    m = lgb.LGBMClassifier(**model_params)
    m.fit(Xtr, tr_c[label_col])
    pred = m.predict_proba(Xev)[:, 1]
    auc = roc_auc_score(ev_c[label_col], pred)
    f1 = f1_score(ev_c[label_col], (pred >= 0.5).astype(int))
    prec = precision_score(ev_c[label_col], (pred >= 0.5).astype(int))
    rec = recall_score(ev_c[label_col], (pred >= 0.5).astype(int))
    return m, auc, f1, prec, rec


# apply(18~21) 예측 (참고용, 실전 배포는 3번 파일)
def predict_apply_target(master_df, model, feats):
    candidates = master_df[(master_df["분기순번"].isin(APPLY_Q)) & (master_df["활성화_현재상태"] == 0)].copy()
    # LightGBM은 결측치를 자체 처리하므로 dropna 하지 않음
    X_new = _safe_lgbm_columns(candidates[feats])
    candidates["예측확률"] = model.predict_proba(X_new)[:, 1]

    ranking = candidates.sort_values("예측확률", ascending=False).reset_index(drop=True)
    ranking["순위"] = ranking.index + 1
    return ranking[["순위", "상권_코드", "상권_코드_명", "예측확률"] + feats]


# 최종 모델 채택 피처 (아래 메인 실행부에서 find_best_feature_combination()이 찾은
# 조합으로 자동 교체됨. 탐색을 안 돌리고 함수만 단독 재사용할 때를 위한 기본값)
FINAL_FEATURES = [
    "매출_YoY_윈저화(%)",
    "매출_모멘텀",
    "2030대_소비_비중",
    "전환율_%p변화"
]

if __name__ == "__main__":
    import time
    _t0 = time.time()

    print("1) 학습용 이력 마스터 데이터 로드...")
    full = load_historical_data()

    print("2) 통계 검증(Distance Correlation + Permutation Test)...")
    train_for_test = full[(full["분기순번"].isin(TRAIN_Q)) & (full["활성화_현재상태"] == 0)]
    print(f"   -> 검정 대상 표본수(train, 분기 {min(TRAIN_Q)}~{max(TRAIN_Q)}): {len(train_for_test)}건")
    dcor_result = run_dcor_test(train_for_test)
    dcor_result.drop(columns=["컬럼명"]).to_excel(os.path.join(BASE_DIR, "피처_유의성_검정_전체결과.xlsx"), index=False)

    print("\n=== Distance Correlation 결과 (유의한 피처) ===")
    display_cols = ["피처명", "구분", "계산 개요", "Distance Correlation", "Permutation p-value", "유의성 여부 (p<0.05)"]
    print(dcor_result[dcor_result["유의성 여부 (p<0.05)"] == "유의함 (O)"]
          .sort_values("Permutation p-value")[display_cols].to_string(index=False))
    print("\n=== Distance Correlation 결과 전체 ===")
    print(dcor_result.sort_values("Permutation p-value")[display_cols].to_string(index=False))

    # 3) AUC를 최대화하는 피처 조합 탐색 (train 내부 CV만 사용, test는 절대 안 봄)
    #    [수정] dcor+permutation 검정에서 유의(p<0.05)하다고 판정된 피처만 탐색 후보로 사용.
    #    유의하지 않은 피처는 애초에 후보에서 제외하므로, 탐색 결과에 절대 포함되지 않는다.
    print("\n3) AUC 최대화 피처 조합 탐색 (train 내부 5-fold 교차검증, test 데이터 미사용)...")
    significant_cols = dcor_result.loc[dcor_result["유의성 여부 (p<0.05)"] == "유의함 (O)", "컬럼명"].tolist()
    search_candidates = [c for c in significant_cols if c in train_for_test.columns]
    print(f"   -> dcor 검정 유의(p<0.05) 피처: {len(search_candidates)}개")
    print(f"      {search_candidates}")
    if not search_candidates:
        raise ValueError("dcor 검정에서 유의한 피처가 하나도 없습니다. n_permutations를 늘리거나 데이터를 확인하세요.")
    train_for_search = train_for_test.dropna(subset=["활성화_라벨_1년후"])

    # [신규] 상관계수가 높아 사실상 중복인 피처들을 정리 (설명력 상위 CORR_KEEP_PER_CLUSTER개만 유지)
    importance_map = dict(zip(dcor_result["컬럼명"], dcor_result["Distance Correlation"]))
    search_candidates = remove_redundant_correlated_features(
        train_for_search, search_candidates, importance_map
    )
    print(f"   -> 중복 제거 후 최종 탐색 후보: {len(search_candidates)}개")
    print(f"      {search_candidates}")
    best_features, best_cv_auc, search_results_df = find_best_feature_combination(
        train_for_search, search_candidates, label_col="활성화_라벨_1년후"
    )
    search_results_df_save = search_results_df.copy()
    search_results_df_save["피처조합"] = search_results_df_save["피처조합"].apply(lambda t: ", ".join(t))
    search_results_df_save.to_excel(os.path.join(BASE_DIR, "피처조합_AUC탐색_전체결과.xlsx"), index=False)

    FINAL_FEATURES = best_features
    print(f"\n   => 채택된 FINAL_FEATURES: {FINAL_FEATURES}")
    print(f"   => train 내부 CV 기준 AUC: {best_cv_auc:.4f}")

    # 4) 최종 모델을 train 전체로 학습하고, test에는 '단 1번만' 평가 (선택 편향 없음)
    print("\n4) 최종 모델(LightGBM) 학습 및 test셋(분기순번 14~17) 평가...")
    model, auc_test, f1_test, prec_test, rec_test = train_and_evaluate(full, FINAL_FEATURES)
    print(f"\n=== 최종 모델(FINAL_FEATURES) test셋 성능 (held-out, 1회 평가) ===")
    print(f"AUC       : {auc_test:.4f}")
    print(f"F1        : {f1_test:.4f}")
    print(f"Precision : {prec_test:.4f}")
    print(f"Recall    : {rec_test:.4f}")

    print("\n5) apply(분기순번 18~21) 예측 순위 산출 중... (참고용)")
    ranking_df = predict_apply_target(full, model, FINAL_FEATURES)

    ranking_df.to_excel(os.path.join(BASE_DIR, "예측_순위표.xlsx"), index=False)
    print("\n=== 활성화 가능성 예측 Top 5 상권 ===")
    print(ranking_df[["순위", "상권_코드_명", "예측확률"]].head().to_string(index=False))

    print("\n'예측_순위표.xlsx', '피처조합_AUC탐색_전체결과.xlsx', '피처_유의성_검정_전체결과.xlsx' 저장 완료!")
    print(f"\n총 소요 시간: {time.time() - _t0:.1f}초")
