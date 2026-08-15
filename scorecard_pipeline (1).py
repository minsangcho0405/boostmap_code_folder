"""
2026년 분기별 예측 스코어카드 생성 파이프라인
================================================
1단계: qcut으로 예측확률 5등급 분류
2단계: 임대료_2026Q2 매핑 (최신 데이터)
3단계: 지속상위후보 스코어를 분기별_통합_요약 기준으로 통일
4단계: 등급별 실제 활성화율 계산 (Lift Chart용)
5단계: 전체 저장
"""

import pandas as pd

# ── 파일 경로 ──────────────────────────────────────────────
SRC       = '2026년_분기별_예측_임대료결합.xlsx'          # 예측 + 임대료 결합 파일
VAL_SRC   = '2026Q1_모델검증_비교데이터.xlsx'             # 실전 검증 파일
RENT_SRC  = '임대동향_지역별_임대료_2024년3분기___소규모_상가.xlsx'
OUTPUT    = '2026년_분기별_예측_스코어카드.xlsx'

# ── 등급 라벨 ──────────────────────────────────────────────
GRADE_LABEL = {
    '1등급': '★★★★★ 최우수',
    '2등급': '★★★★☆ 우수',
    '3등급': '★★★☆☆ 양호',
    '4등급': '★★☆☆☆ 보통',
    '5등급': '★☆☆☆☆ 관심',
}

# ── STEP 0: 임대료 2026Q2 매핑 테이블 ─────────────────────
df_rent = pd.read_excel(RENT_SRC)
df_rent = df_rent.iloc[2:].reset_index(drop=True)
df_rent.columns = [
    'No','지역대','지역중','지역소',
    '2024Q3','2024Q4','2025Q1','2025Q2',
    '2025Q3','2025Q4','2026Q1','2026Q2'
]
df_rent = df_rent[df_rent['지역소'].notna() & (df_rent['지역소'] != '지역')].copy()
df_rent['임대료_2026Q2'] = pd.to_numeric(df_rent['2026Q2'], errors='coerce')
rent_map = dict(zip(df_rent['지역소'], df_rent['임대료_2026Q2']))

# ── STEP 1: 분기별_통합_요약에서 스코어 생성 ──────────────
xl = pd.ExcelFile(SRC)
통합 = pd.read_excel(SRC, sheet_name='분기별_통합_요약')
통합['임대료_2026Q2'] = 통합['임대료_지역'].map(rent_map).fillna('--')
통합['스코어등급'] = pd.qcut(
    통합['평균예측확률(%)'], q=5,
    labels=['5등급','4등급','3등급','2등급','1등급'],
    duplicates='drop'
)
통합['등급_라벨'] = 통합['스코어등급'].map(GRADE_LABEL)

# ── STEP 2: 지속상위후보 — 통합요약 스코어 그대로 매핑 ────
# (지속후보는 통합요약의 부분집합이므로 등급을 통일)
지속 = pd.read_excel(SRC, sheet_name='지속상위후보')
지속['임대료_2026Q2'] = 지속['임대료_지역'].map(rent_map).fillna('--')
지속 = 지속.merge(
    통합[['상권_코드','스코어등급','등급_라벨']],
    on='상권_코드', how='left'
)

# ── STEP 3: 분기별 예측 시트 등급 부여 ────────────────────
def add_grade(df, prob_col):
    """예측확률 높을수록 1등급"""
    df['스코어등급'] = pd.qcut(
        df[prob_col], q=5,
        labels=['5등급','4등급','3등급','2등급','1등급'],
        duplicates='drop'
    )
    df['등급_라벨'] = df['스코어등급'].map(GRADE_LABEL)
    return df

분기_시트 = [
    '2025Q1_예측(2026Q1대상)',
    '2025Q2_예측(2026Q2대상)',
    '2025Q3_예측(2026Q3대상)',
    '2025Q4_예측(2026Q4대상)',
]
sheet_dfs = {}
for sh in 분기_시트:
    df = pd.read_excel(SRC, sheet_name=sh)
    df['임대료_2026Q2'] = df['임대료_지역'].map(rent_map).fillna('--')
    if '임대료_2026Q1' in df.columns:
        df = df.drop(columns=['임대료_2026Q1'])
    df = add_grade(df, '예측확률')
    sheet_dfs[sh] = df

sheet_dfs['분기별_통합_요약'] = 통합
sheet_dfs['지속상위후보'] = 지속

# ── STEP 4: 등급별 실제 활성화율 계산 (2025Q1 → 2026Q1 검증) ──
val = pd.read_excel(VAL_SRC, sheet_name='예측_vs_실제_2026Q1')[
    ['상권_코드', '활성화_실제_2026Q1']
]
q1 = sheet_dfs['2025Q1_예측(2026Q1대상)'].merge(val, on='상권_코드', how='left')
q1_valid = q1.dropna(subset=['활성화_실제_2026Q1', '스코어등급'])
기저율 = q1_valid['활성화_실제_2026Q1'].mean() * 100

grade_summary = q1_valid.groupby('스코어등급', observed=True).agg(
    상권수=('상권_코드', 'count'),
    실제활성화수=('활성화_실제_2026Q1', 'sum'),
    실제활성화율=('활성화_실제_2026Q1', 'mean'),
    평균예측확률=('예측확률', 'mean'),
).reset_index()
grade_summary['실제활성화율(%)'] = (grade_summary['실제활성화율'] * 100).round(1)
grade_summary['Lift']           = (grade_summary['실제활성화율(%)'] / 기저율).round(2)
grade_summary['평균예측확률(%)'] = grade_summary['평균예측확률'].round(1)
grade_summary['등급_라벨']       = grade_summary['스코어등급'].map(GRADE_LABEL)
grade_summary = grade_summary[[
    '스코어등급','등급_라벨','상권수','실제활성화수',
    '실제활성화율(%)','Lift','평균예측확률(%)'
]]

print(f"기저율: {기저율:.1f}%")
print(grade_summary.to_string(index=False))

# ── STEP 5: 저장 ───────────────────────────────────────────
with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
    for sh, df in sheet_dfs.items():
        df.to_excel(writer, sheet_name=sh, index=False)
    grade_summary.to_excel(writer, sheet_name='등급별_활성화율_검증', index=False)

print(f"\n저장 완료 → {OUTPUT}")
