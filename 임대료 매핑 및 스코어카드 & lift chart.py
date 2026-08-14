# -*- coding: utf-8 -*-
"""
==============================================================================
소비·방문 패턴 변화 기반 서울시 외식 상권 조기 예측 모델 — 최종 코드 통합본
==============================================================================
이 스크립트는 프로젝트에서 최종적으로 채택한 방법론만을 반영한 end-to-end
파이프라인이다. (탐색 과정에서 검토 후 기각한 대안 라벨/피처 정의는 제외)

[최종 확정 사항]
1. 활성화 라벨: 직전4분기 매출합 하위25% 제외(최소규모필터)
                + 누적4분기 매출성장률 상위20%(CAGR)
2. 최종 피처(6개): 매출_저점대비_반등폭, 조건_동시충족_최근4분기_충족횟수,
                  전환율_%p변화, 구매전환율_100cap(%),
                  분기별_총_유동인구_수, 저녁심야_매출_비중(%)
3. 데이터 분할: 분기순번 기준 롤링 윈도우
                train(8~11) / validation(12~13) / test(14~16) / apply(17~)
4. 모델: 로지스틱 회귀(class_weight='balanced'), StandardScaler 표준화

[입력 파일]
- 데이터_전처리_최종_변수_추가.xlsx : 2021Q1~2025Q4 상권×분기 매출/유동인구 기초 데이터
- 매출_건수_추가.xlsx               : 2021Q1~2025Q4 매출건수 데이터
- 서울시_상권분석서비스_추정매출-상권_.csv    : 신규 분기 원본 매출 데이터(예: 2026Q1)
- 서울시_상권분석서비스_길단위인구-상권_.csv  : 신규 분기 원본 유동인구 데이터

[출력 파일]
- 최종_모델링데이터.xlsx        : 라벨+피처 결합 최종 데이터
- t검정_결과.xlsx               : 18개 피처 통계 검증 결과
- 모델_평가_결과.xlsx           : 모델 A/B 비교 및 test 최종 평가
- 예측_순위표.xlsx              : 신규 분기 예측 순위표
==============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import joblib

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# 최종 확정 피처 목록 (전체 파이프라인에서 공통 사용)
FINAL_FEATURES = [
    "매출_저점대비_반등폭",
    "조건_동시충족_최근4분기_충족횟수",
    "전환율_%p변화",
    "구매전환율_100cap(%)",
    "분기별_총_유동인구_수",
    "저녁심야_매출_비중(%)",
]

FOOD10 = ["한식음식점", "중식음식점", "일식음식점", "양식음식점", "분식전문점",
          "패스트푸드점", "치킨전문점", "제과점", "커피-음료", "호프-간이주점"]
DISTRICTS = ["골목상권", "발달상권"]


# =============================================================================
# 유틸리티 함수
# =============================================================================
def safe_shift(df, group_col_name, value_col, n):
    """상권별로 n분기 전 값을 정확한 분기 간격 확인 후 반환 (분기 gap 안전 처리)"""
    shifted_val = df.groupby(group_col_name)[value_col].shift(n)
    shifted_qtr = df.groupby(group_col_name)["분기순번"].shift(n)
    valid = (df["분기순번"] - shifted_qtr) == n
    return shifted_val.where(valid)


def cohens_d(a, b):
    """Welch 기준 효과크기(pooled sd)"""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def add_quarters(yq, n):
    """YYYYQ 형식 분기코드에 n분기를 더함(n은 음수 가능)"""
    year, q = yq // 10, yq % 10
    total_q = (year * 4 + (q - 1)) + n
    return (total_q // 4) * 10 + (total_q % 4) + 1


# =============================================================================
# STEP 1. 기존 이력 데이터(2021Q1~2025Q4) 로드
#   -> 이미 검증을 마친 최종 데이터 파일을 그대로 사용한다.
#      (라벨/피처 계산 로직은 STEP 3, 4 함수에 그대로 보존되어 있으며,
#       신규 분기 데이터 처리에 동일하게 재사용된다. 과거 이력에 대해서는
#       프로젝트 진행 중 여러 차례(원본 대조 2회 포함) 검증을 마친
#       '데이터_최종_활성화라벨_CAGR규모필터.xlsx'를 신뢰 소스로 사용한다.)
# =============================================================================
def load_historical_data():
    """검증 완료된 최종 데이터(분기순번 1~20)를 로드한다."""
    master = pd.read_excel("데이터_최종_활성화라벨_CAGR규모필터.xlsx", sheet_name="최종_모델링데이터")
    master = master.rename(columns={
        "활성화_현재상태(CAGR+규모필터)_0또는1": "활성화_현재상태",
        "활성화_라벨_1년후(CAGR+규모필터)_0또는1": "활성화_라벨_1년후",
        "매출_YoY증가율_윈저화(%)": "매출_YoY_윈저화(%)",
    })
    return master


# =============================================================================
# STEP 2. 신규 분기 원본 데이터 처리 (원본 CSV -> 상권×분기 집계)
# =============================================================================
def build_new_quarter_data(revenue_csv_path, foot_csv_path, target_yq, quarter_seq):
    """서울시 상권분석서비스 원본 CSV로부터 신규 분기 데이터를 만든다."""
    rev_raw = pd.read_csv(revenue_csv_path, encoding="cp949")
    foot_raw = pd.read_csv(foot_csv_path, encoding="cp949")

    rev_f = rev_raw[(rev_raw["기준_년분기_코드"] == target_yq)
                     & rev_raw["서비스_업종_코드_명"].isin(FOOD10)
                     & rev_raw["상권_구분_코드_명"].isin(DISTRICTS)]
    agg_cols = ["당월_매출_금액", "당월_매출_건수", "시간대_17~21_매출_금액", "시간대_21~24_매출_금액"]
    agg = rev_f.groupby(["상권_코드", "상권_코드_명"], as_index=False)[agg_cols].sum()

    foot_f = foot_raw[(foot_raw["기준_년분기_코드"] == target_yq)
                       & foot_raw["상권_구분_코드_명"].isin(DISTRICTS)][["상권_코드", "총_유동인구_수"]]

    new = agg.merge(foot_f, on="상권_코드", how="inner")  # 매출·유동인구 모두 확보된 상권만 (안전한 방식)
    new["기준_년분기_코드"] = target_yq
    new["분기순번"] = quarter_seq
    new["저녁심야_매출_비중(%)"] = (
        (new["시간대_17~21_매출_금액"] + new["시간대_21~24_매출_금액"]) / new["당월_매출_금액"] * 100
    )
    new = new.rename(columns={
        "당월_매출_금액": "외식업_당월_매출_금액",
        "당월_매출_건수": "외식업_총_매출_건수",
        "총_유동인구_수": "분기별_총_유동인구_수",
    })
    return new[["상권_코드", "상권_코드_명", "기준_년분기_코드", "분기순번",
                "외식업_당월_매출_금액", "외식업_총_매출_건수",
                "분기별_총_유동인구_수", "저녁심야_매출_비중(%)"]]


# =============================================================================
# STEP 3. 활성화 라벨 계산 (CAGR + 최소규모필터)
# =============================================================================
def compute_activation_label(df):
    """
    라벨 정의: 직전4분기 매출합 하위25% 제외(최소규모필터)
              + 누적4분기 매출성장률 상위20%(CAGR)
    현재상태(t)와 1년후 라벨(t+4) 모두 동일 정의로 계산한다.
    """
    df = df.sort_values(["상권_코드", "분기순번"]).reset_index(drop=True)
    g = df.groupby("상권_코드")

    rev_recent4 = sum(g["외식업_당월_매출_금액"].shift(k) for k in range(4))
    rev_prior4 = sum(g["외식업_당월_매출_금액"].shift(4 + k) for k in range(4))
    df["직전4분기_매출합"] = rev_prior4
    df["누적4분기_성장률(%)"] = np.where(rev_prior4 > 0, (rev_recent4 - rev_prior4) / rev_prior4 * 100, np.nan)

    df["규모필터_통과"] = df.groupby("기준_년분기_코드")["직전4분기_매출합"].transform(
        lambda x: x >= x.quantile(0.25) if x.notna().sum() > 0 else False
    )

    df["활성화_현재상태"] = np.nan
    for qtr, idx in df.groupby("기준_년분기_코드").groups.items():
        sub = df.loc[idx]
        valid = sub["규모필터_통과"] & sub["누적4분기_성장률(%)"].notna()
        if valid.sum() == 0:
            continue
        cutoff = sub.loc[valid, "누적4분기_성장률(%)"].quantile(0.80)
        result = (sub["누적4분기_성장률(%)"] >= cutoff) & valid
        df.loc[idx, "활성화_현재상태"] = result.astype(float).where(
            sub["누적4분기_성장률(%)"].notna() & sub["규모필터_통과"]
        )

    # t+4 시점 라벨 결합 (1년 후 결과)
    future_active = df.groupby("상권_코드")["활성화_현재상태"].shift(-4)
    future_qtr = df.groupby("상권_코드")["분기순번"].shift(-4)
    valid_future = (future_qtr - df["분기순번"]) == 4
    df["활성화_라벨_1년후"] = np.where(valid_future, future_active, np.nan)

    return df


# =============================================================================
# STEP 4. 최종 6개 피처 계산
# =============================================================================
def compute_final_features(df, rev_col, cnt_col, foot_col):
    """
    최종 확정 6개 피처를 계산한다.
    - 매출_저점대비_반등폭 : 매출_YoY 기반 (t vs 최근3분기 최저치)
    - 조건_동시충족_최근4분기_충족횟수 : 유동인구증가 & 전환율증가 동시충족 최근4분기 합
    - 전환율_%p변화 : 구매전환율(t) - 구매전환율(t-4)
    - 구매전환율_100cap(%) : 매출건수/유동인구*100, 100% 상한
    - 분기별_총_유동인구_수 : 원본 그대로
    - 저녁심야_매출_비중(%) : 원본 그대로 (STEP1/2에서 이미 계산됨)
    """
    df = df.sort_values(["상권_코드", "분기순번"]).reset_index(drop=True)

    # --- 유동인구_YoY, 전환율_%p변화, 구매전환율_100cap : 원천 데이터로 전 구간 직접 계산 ---
    fp_t4 = safe_shift(df, "상권_코드", foot_col, 4)
    df["유동인구_YoY(%)"] = np.where(fp_t4 > 0, (df[foot_col] / fp_t4 - 1) * 100, np.nan)

    conv_now = np.minimum(df[cnt_col] / df[foot_col] * 100, 100)
    df["구매전환율_100cap(%)"] = conv_now
    df["_conv_temp"] = conv_now
    conv_t4 = safe_shift(df, "상권_코드", "_conv_temp", 4)
    df["전환율_%p변화"] = df["_conv_temp"] - conv_t4
    df = df.drop(columns=["_conv_temp"])

    # --- 조건_동시충족_최근4분기_충족횟수 ---
    df["조건_동시충족"] = ((df["유동인구_YoY(%)"] > 0) & (df["전환율_%p변화"] > 0)).astype(float)
    df.loc[df["유동인구_YoY(%)"].isna() | df["전환율_%p변화"].isna(), "조건_동시충족"] = np.nan

    def rolling4_count(df, value_col):
        result = pd.Series(np.nan, index=df.index)
        for code, idx in df.groupby("상권_코드").groups.items():
            sub = df.loc[idx].sort_values("분기순번")
            qtrs, vals = sub["분기순번"].values, sub[value_col].values
            counts = np.full(len(sub), np.nan)
            for i in range(len(sub)):
                w_idx = [j for j in range(len(sub)) if qtrs[i] - 3 <= qtrs[j] <= qtrs[i]]
                w_qtrs = qtrs[w_idx]
                if len(w_qtrs) == 4 and (w_qtrs.max() - w_qtrs.min() == 3):
                    w = vals[w_idx]
                    if not np.isnan(w).any():
                        counts[i] = w.sum()
            result.loc[sub.index] = counts
        return result

    df["조건_동시충족_최근4분기_충족횟수"] = rolling4_count(df, "조건_동시충족")

    # --- 매출_YoY(윈저화) 및 매출_저점대비_반등폭 ---
    rev_t4 = safe_shift(df, "상권_코드", rev_col, 4)
    df["매출_YoY_원본(%)"] = np.where(rev_t4 > 0, (df[rev_col] / rev_t4 - 1) * 100, np.nan)
    # 분기별 횡단면 1~99 percentile 윈저화
    df["매출_YoY_윈저화(%)"] = df.groupby("기준_년분기_코드")["매출_YoY_원본(%)"].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
    )

    yoy = df["매출_YoY_윈저화(%)"]
    yoy_t1 = safe_shift(df, "상권_코드", "매출_YoY_윈저화(%)", 1)
    yoy_t2 = safe_shift(df, "상권_코드", "매출_YoY_윈저화(%)", 2)
    min_recent3 = pd.concat([yoy, yoy_t1, yoy_t2], axis=1).min(axis=1, skipna=False)
    df["매출_저점대비_반등폭"] = yoy - min_recent3

    return df


# =============================================================================
# STEP 5. data_split 부여 (분기순번 기준 롤링 윈도우)
# =============================================================================
def assign_data_split(df):
    def split(row):
        if pd.isna(row["활성화_현재상태"]):
            return "제외(feature결측)"
        q = row["분기순번"]
        if 8 <= q <= 11:
            return "train"
        elif 12 <= q <= 13:
            return "validation"
        elif 14 <= q <= 16:
            return "test"
        elif q >= 17:
            return "apply_예측대상"
        return "제외(범위밖)"
    df["data_split"] = df.apply(split, axis=1)
    return df


# =============================================================================
# STEP 6. t-검정 (18개 후보 피처 통계 검증) — 최종 확정 6개 포함 전체 후보
# =============================================================================
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


# =============================================================================
# STEP 7. 모델 학습 (A: 매출YoY만 vs B: 최종 6개 피처) 및 평가
# =============================================================================
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

    # 모델 A (기준선) vs 모델 B (최종안) - validation 평가
    _, _, auc_a, f1_a = fit_eval(["매출_YoY_윈저화(%)"], train, val)
    model_b, scaler_b, auc_b, f1_b = fit_eval(FINAL_FEATURES, train, val)

    print(f"모델 A (매출YoY만)   : Val AUC={auc_a:.3f}, F1={f1_a:.3f}")
    print(f"모델 B (최종 6개피처) : Val AUC={auc_b:.3f}, F1={f1_b:.3f}")

    # 최종 test 1회 평가 (공식 보고 수치) — train 구간만으로 학습한 모델을 그대로 사용
    # (validation은 이미 위에서 튜닝/비교 용도로만 썼고, test 평가 시점에는 재활용하지 않음 — 데이터 유출 방지)
    _, _, auc_test_official, f1_test_official = fit_eval(FINAL_FEATURES, train, test)
    tr_c = train[FINAL_FEATURES + [label_col]].dropna()
    te_c = test[FINAL_FEATURES + [label_col]].dropna()
    sc_official = StandardScaler()
    Xtr_o = sc_official.fit_transform(tr_c[FINAL_FEATURES])
    Xte_o = sc_official.transform(te_c[FINAL_FEATURES])
    model_official = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr_o, tr_c[label_col])
    pred_o = model_official.predict_proba(Xte_o)[:, 1]
    precision_o = precision_score(te_c[label_col], (pred_o >= 0.5).astype(int))
    recall_o = recall_score(te_c[label_col], (pred_o >= 0.5).astype(int))
    print(f"모델 B test 최종 평가(공식, train만 학습) : AUC={auc_test_official:.3f}, F1={f1_test_official:.3f}, "
          f"Precision={precision_o:.3f}, Recall={recall_o:.3f}")

    # 참고용: 배포 목적 재학습(train+validation 결합) — 검증 지표가 아닌 실전 배포용 별도 모델
    # 공식 test 성능(위)과는 목적이 다르므로 수치가 미세하게 달라질 수 있음(정상)
    train_plus_val = pd.concat([train, val])
    _, _, auc_deploy, f1_deploy = fit_eval(FINAL_FEATURES, train_plus_val, test)
    print(f"(참고, 비공식) 배포용 재학습(train+val 결합) test 성능 : AUC={auc_deploy:.3f}, F1={f1_deploy:.3f}")

    return model_official, sc_official


# =============================================================================
# STEP 7-1. 학습된 모델 저장 / 불러오기
# =============================================================================
def save_model(model, scaler, path_prefix="final_model"):
    """학습된 로지스틱 회귀 모델과 스케일러를 파일로 저장 (재학습 없이 재사용 가능)"""
    joblib.dump(model, f"{path_prefix}.pkl")
    joblib.dump(scaler, f"{path_prefix}_scaler.pkl")
    print(f"모델 저장 완료: {path_prefix}.pkl, {path_prefix}_scaler.pkl")


def load_model(path_prefix="final_model"):
    """저장된 모델과 스케일러를 불러온다."""
    model = joblib.load(f"{path_prefix}.pkl")
    scaler = joblib.load(f"{path_prefix}_scaler.pkl")
    return model, scaler


# =============================================================================
# STEP 8. 신규 분기 예측 순위표 생성
# =============================================================================
def predict_ranking(df, model, scaler, target_split="apply_예측대상"):
    candidates = df[(df["data_split"] == target_split) & (df["활성화_현재상태"] == 0)].copy()
    candidates = candidates.dropna(subset=FINAL_FEATURES)
    X = scaler.transform(candidates[FINAL_FEATURES])
    candidates["예측확률"] = model.predict_proba(X)[:, 1]
    ranking = candidates.sort_values("예측확률", ascending=False).reset_index(drop=True)
    ranking["순위"] = ranking.index + 1
    return ranking[["순위", "상권_코드", "상권_코드_명", "예측확률"] + FINAL_FEATURES]


# =============================================================================
# MAIN — 파이프라인 실행
# =============================================================================
if __name__ == "__main__":
    # 1) 기존 이력 로드 (라벨·피처·data_split 모두 이미 계산되어 검증된 상태)
    full = load_historical_data()

    # 1-1) 신규 분기가 있는 경우: 원본 CSV로부터 만들고, 동일 로직으로 라벨/피처 계산 후 결합
    #      (아래는 예시 — 신규 분기 파일이 있을 때만 주석 해제)
    # new_q = build_new_quarter_data("서울시_상권분석서비스_추정매출-상권_.csv",
    #                                  "서울시_상권분석서비스_길단위인구-상권_.csv",
    #                                  target_yq=20261, quarter_seq=21)
    # combined = pd.concat([full, new_q], ignore_index=True)
    # combined = compute_activation_label(combined)      # 활성화_현재상태/1년후 재계산 (전 구간)
    # combined = compute_final_features(combined, rev_col="외식업_당월_매출_금액",
    #                                    cnt_col="외식업_총_매출_건수", foot_col="분기별_총_유동인구_수")
    # combined = assign_data_split(combined)
    # full = combined

    # 2) t-검정 (train 구간, 현재 비활성화 상권 대상)
    train_for_ttest = full[(full["data_split"] == "train") & (full["활성화_현재상태"] == 0)]
    ttest_result = run_ttest(train_for_ttest)
    print(ttest_result.to_string(index=False))
    ttest_result.to_excel("t검정_결과.xlsx", index=False)
    print()

    # 3) 모델 학습 및 평가 (모델 A vs B, test 최종 1회 평가)
    model, scaler = train_and_evaluate(full)

    # 3-1) 배포용 최종 모델: 라벨 보유 전체 데이터(train+validation+test)로 재학습 후 저장
    #      (test 평가는 3번에서 이미 1회 완료 — 이후 이 모델로 실제 예측에 사용)
    all_labeled = full[
        (full["data_split"].isin(["train", "validation", "test"])) & (full["활성화_현재상태"] == 0)
    ]
    tr_all = all_labeled[FINAL_FEATURES + ["활성화_라벨_1년후"]].dropna()
    deploy_scaler = StandardScaler()
    X_all = deploy_scaler.fit_transform(tr_all[FINAL_FEATURES])
    deploy_model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(
        X_all, tr_all["활성화_라벨_1년후"]
    )
    save_model(deploy_model, deploy_scaler, path_prefix="final_model")

    # 4) (신규 분기 있을 경우) 저장된 배포용 모델로 예측 순위표 생성
    # deploy_model, deploy_scaler = load_model("final_model")  # 재학습 없이 불러와 바로 사용 가능
    # ranking = predict_ranking(full, deploy_model, deploy_scaler)
    # ranking.to_excel("예측_순위표.xlsx", index=False)

    # 5) 최종 데이터 저장
    full.to_excel("최종_모델링데이터_재현.xlsx", index=False)
    print("\n파이프라인 실행 완료.")
