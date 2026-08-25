# -*- coding: utf-8 -*-
# [평가용] 최종 피처 모델 - AUC/F1 성능 측정용 (train만 학습, test는 순수 평가)
import os
# ==== 출력 파일 저장 경로 지정 ====
OUTPUT_DIR = r"C:\your_file_path" # 파일 저장 경로 작성
os.makedirs(OUTPUT_DIR, exist_ok=True)   # 폴더 없으면 자동 생성
os.chdir(OUTPUT_DIR)                     # 이후 모든 상대경로 저장은 여기 기준

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    target_file = os.path.join(BASE_DIR, "데이터_최종_활성화라벨_CAGR규모필터_6.xlsx")
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
            "피처명": label_kr, "구분": 구분, "계산 개요": 설명,
            "Distance Correlation": round(dcor, 4),
            "Permutation p-value": round(p_val, 3),
            "유의성 여부 (p<0.05)": "유의함 (O)" if p_val < 0.05 else "유의하지 않음 (X)",
        })
    return pd.DataFrame(results)


# train(8~13) 학습, test(14~17) 평가
def train_and_evaluate(df, label_col="활성화_라벨_1년후"):
    train = df[(df["분기순번"].isin(TRAIN_Q)) & (df["활성화_현재상태"] == 0)]
    test = df[(df["분기순번"].isin(TEST_Q)) & (df["활성화_현재상태"] == 0)]

    def fit_eval(feats, tr, ev):
        tr_c = tr[feats + [label_col]].dropna()
        ev_c = ev[feats + [label_col]].dropna()
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr_c[feats])
        Xev = sc.transform(ev_c[feats])
        m = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42).fit(Xtr, tr_c[label_col])
        pred = m.predict_proba(Xev)[:, 1]
        auc = roc_auc_score(ev_c[label_col], pred)
        f1 = f1_score(ev_c[label_col], (pred >= 0.5).astype(int))
        return m, sc, auc, f1

    model_official, sc_official, auc_test_official, f1_test_official = fit_eval(FINAL_FEATURES, train, test)
    return model_official, sc_official, auc_test_official, f1_test_official


# apply(18~21) 예측 (참고용, 실전 배포는 3번 파일)
def predict_apply_target(master_df, model, scaler):
    candidates = master_df[(master_df["분기순번"].isin(APPLY_Q)) & (master_df["활성화_현재상태"] == 0)].copy()
    candidates = candidates.dropna(subset=FINAL_FEATURES)

    X_new = scaler.transform(candidates[FINAL_FEATURES])
    candidates["예측확률"] = model.predict_proba(X_new)[:, 1]

    ranking = candidates.sort_values("예측확률", ascending=False).reset_index(drop=True)
    ranking["순위"] = ranking.index + 1
    return ranking[["순위", "상권_코드", "상권_코드_명", "예측확률"] + FINAL_FEATURES]


# 최종 모델 채택 피처
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

    print("2) 통계 검증(Distance Correlation + Permutation Test) 및 모델 학습...")
    train_for_test = full[(full["분기순번"].isin(TRAIN_Q)) & (full["활성화_현재상태"] == 0)]
    print(f"   -> 검정 대상 표본수(train, 분기 {min(TRAIN_Q)}~{max(TRAIN_Q)}): {len(train_for_test)}건")
    dcor_result = run_dcor_test(train_for_test)
    dcor_result.to_excel(os.path.join(BASE_DIR, "피처_유의성_검정_전체결과.xlsx"), index=False)

    print("\n=== Distance Correlation 결과 (유의한 피처) ===")
    print(dcor_result[dcor_result["유의성 여부 (p<0.05)"] == "유의함 (O)"]
          .sort_values("Permutation p-value").to_string(index=False))
    print("\n=== Distance Correlation 결과 전체 ===")
    print(dcor_result.sort_values("Permutation p-value").to_string(index=False))

    model, scaler, auc_test, f1_test = train_and_evaluate(full)
    print(f"\n=== 최종 모델(FINAL_FEATURES) test셋(분기순번 14~17) 성능 ===")
    print(f"AUC: {auc_test:.4f}")
    print(f"F1 : {f1_test:.4f}")

    print("3) apply(분기순번 18~21) 예측 순위 산출 중... (참고용)")
    ranking_df = predict_apply_target(full, model, scaler)

    ranking_df.to_excel(os.path.join(BASE_DIR, "예측_순위표.xlsx"), index=False)
    print("\n=== 활성화 가능성 예측 Top 5 상권 ===")
    print(ranking_df[["순위", "상권_코드_명", "예측확률"]].head().to_string(index=False))

    print("\n'예측_순위표.xlsx' 저장 완료!")
    print(f"\n총 소요 시간: {time.time() - _t0:.1f}초")