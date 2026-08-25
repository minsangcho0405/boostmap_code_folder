-- 1. 기초 매출데이터에서 대상 업종 및 상권유형 필터링 (트렌디 업종 구분용 컬럼 포함)[cite: 1, 2]
WITH filtered_raw AS (
    SELECT
        상권_코드,
        상권_코드_명,
        기준_년분기_코드,
        당월_매출_금액,
        당월_매출_건수,
        연령대_20_매출_금액,
        연령대_30_매출_금액,
        `시간대_17~21_매출_금액`,
        `시간대_21~24_매출_금액`,
        주말_매출_금액,
        서비스_업종_코드_명,
        CASE WHEN 서비스_업종_코드_명 IN ('커피-음료', '제과점', '패스트푸드점') THEN 당월_매출_금액 ELSE 0 END AS 트렌디_매출_금액
    FROM raw_sales
    WHERE 서비스_업종_코드_명 IN (
        '한식음식점','중식음식점','일식음식점','양식음식점','분식전문점',
        '패스트푸드점','호프-간이주점','치킨전문점','커피-음료','제과점'
    )
    AND 상권_구분_코드_명 IN ('골목상권', '발달상권')
),

-- 2. 상권/분기 단위 매출 및 연령대·업종별 집계[cite: 1]
district_quarter AS (
    SELECT
        상권_코드,
        MAX(상권_코드_명) AS 상권_코드_명,
        기준_년분기_코드,
        SUM(당월_매출_금액) AS 매출_금액,
        SUM(당월_매출_건수) AS 매출_건수,
        SUM(연령대_20_매출_금액 + 연령대_30_매출_금액) AS 매출_2030,
        SUM(`시간대_17~21_매출_금액` + `시간대_21~24_매출_금액`) AS 매출_저녁심야,
        SUM(주말_매출_금액) AS 매출_주말,
        SUM(트렌디_매출_금액) AS 매출_트렌디
    FROM filtered_raw
    GROUP BY 상권_코드, 기준_년분기_코드
),

-- 3. 연도/분기 코드를 시계열 정렬용 순차 번호로 변환[cite: 1]
with_seq AS (
    SELECT
        dq.*,
        (FLOOR(기준_년분기_코드 / 10) - 2021) * 4
            + MOD(기준_년분기_코드, 10) AS 분기순번
    FROM district_quarter dq
),

-- 4. 유동인구 테이블 결합[cite: 1]
with_footfall AS (
    SELECT
        w.*,
        f.총_유동인구_수 AS 분기별_총_유동인구_수
    FROM with_seq w
    LEFT JOIN raw_footfall f
        ON w.상권_코드 = f.상권_코드
       AND w.기준_년분기_코드 = f.기준_년분기_코드
),

-- 5. 매출 YoY 계산 및 전년 동기 유동인구 매핑[cite: 1, 2]
with_yoy AS (
    SELECT
        w.*,
        LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) AS 전년동기매출,
        LAG(분기별_총_유동인구_수, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) AS 전년동기_유동인구,
        CASE
            WHEN LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) IS NULL THEN NULL
            WHEN LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) < 5000000 THEN NULL
            ELSE (매출_금액 - LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번))
                 / NULLIF(LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번), 0) * 100
        END AS 매출_YoY_원본
    FROM with_footfall w
),

-- 6. 매출 YoY 이상치 제거용 상/하위 백분위 계산[cite: 1]
pctile_bounds AS (
    SELECT 기준_년분기_코드, 매출_YoY_원본,
           PERCENT_RANK() OVER (PARTITION BY 기준_년분기_코드 ORDER BY 매출_YoY_원본) AS pr
    FROM with_yoy
    WHERE 매출_YoY_원본 IS NOT NULL
),
p1_p99 AS (
    SELECT 기준_년분기_코드,
        MAX(CASE WHEN pr <= 0.01 THEN 매출_YoY_원본 END) AS p1,
        MIN(CASE WHEN pr >= 0.99 THEN 매출_YoY_원본 END) AS p99
    FROM pctile_bounds
    GROUP BY 기준_년분기_코드
),

-- 7. 매출 YoY 1~99% 윈저화 처리[cite: 1]
with_winsorized AS (
    SELECT
        w.*,
        CASE
            WHEN w.매출_YoY_원본 IS NULL THEN NULL
            WHEN w.매출_YoY_원본 < b.p1 THEN b.p1
            WHEN w.매출_YoY_원본 > b.p99 THEN b.p99
            ELSE w.매출_YoY_원본
        END AS 매출_YoY_윈저화
    FROM with_yoy w
    LEFT JOIN p1_p99 b ON w.기준_년분기_코드 = b.기준_년분기_코드
),

-- 8. 최근 4분기 매출 합계 계산[cite: 1]
with_rolling AS (
    SELECT
        w.*,
        SUM(매출_금액) OVER (
            PARTITION BY 상권_코드 ORDER BY 분기순번
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS 최근4분기_매출합
    FROM with_winsorized w
),

-- 9. 직전 4분기 매출 합계 매핑[cite: 1]
with_prev4 AS (
    SELECT
        w.*,
        LAG(최근4분기_매출합, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) AS 직전4분기_매출합
    FROM with_rolling w
),

-- 10. 4분기 누적 CAGR 연간 성장률 계산[cite: 1]
with_cagr AS (
    SELECT
        w.*,
        CASE
            WHEN 직전4분기_매출합 IS NULL OR 직전4분기_매출합 = 0 THEN NULL
            ELSE (최근4분기_매출합 - 직전4분기_매출합) / 직전4분기_매출합 * 100
        END AS 누적4분기_성장률
    FROM with_prev4 w
    WHERE 분기순번 >= 8
),

-- 11. 백분위 순위 산출 (규모 및 성장률)[cite: 1]
with_rank AS (
    SELECT
        w.*,
        PERCENT_RANK() OVER (PARTITION BY 기준_년분기_코드 ORDER BY 직전4분기_매출합) AS 규모_pr,
        PERCENT_RANK() OVER (PARTITION BY 기준_년분기_코드 ORDER BY 누적4분기_성장률) AS cagr_pr
    FROM with_cagr w
    WHERE 직전4분기_매출합 IS NOT NULL AND 누적4분기_성장률 IS NOT NULL
),

-- 12. 활성화 상태 및 타겟 라벨 산출[cite: 1]
with_label AS (
    SELECT
        w.*,
        CASE WHEN 규모_pr >= 0.25 THEN 1 ELSE 0 END AS 규모필터_통과여부,
        CASE WHEN 규모_pr >= 0.25 AND cagr_pr >= 0.80 THEN 1 ELSE 0 END AS 활성화_현재상태
    FROM with_rank w
),

-- 13. 기본 파생 피처 계산 (비중, 전환율, 모멘텀 등)[cite: 1, 2]
with_features AS (
    SELECT
        w.*,
        (w.매출_YoY_윈저화 - LEAST(
            COALESCE(LAG(w.매출_YoY_윈저화, 1) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번), w.매출_YoY_윈저화),
            COALESCE(LAG(w.매출_YoY_윈저화, 2) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번), w.매출_YoY_윈저화),
            COALESCE(LAG(w.매출_YoY_윈저화, 3) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번), w.매출_YoY_윈저화)
        )) AS 매출_저점대비_반등폭,

        (
            (w.매출_YoY_윈저화 + COALESCE(LAG(w.매출_YoY_윈저화, 1) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번), w.매출_YoY_윈저화)) / 2
          -
            (COALESCE(LAG(w.매출_YoY_윈저화, 2) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번), w.매출_YoY_윈저화)
             + COALESCE(LAG(w.매출_YoY_윈저화, 3) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번), w.매출_YoY_윈저화)) / 2
        ) AS 매출_모멘텀,

        CASE
            WHEN w.분기별_총_유동인구_수 IS NULL OR w.분기별_총_유동인구_수 = 0 THEN NULL
            ELSE LEAST(w.매출_건수 / w.분기별_총_유동인구_수 * 100, 100)
        END AS 구매전환율_100cap,

        CASE WHEN w.매출_금액 = 0 THEN NULL ELSE w.매출_저녁심야 / w.매출_금액 * 100 END AS 저녁심야_매출_비중,
        CASE WHEN w.매출_금액 = 0 THEN NULL ELSE w.매출_주말 / w.매출_금액 * 100 END AS 주말_매출_비중,
        CASE WHEN w.매출_금액 = 0 THEN NULL ELSE w.매출_2030 / w.매출_금액 END AS `2030대_소비_비중`,
        CASE WHEN w.매출_금액 = 0 THEN NULL ELSE w.매출_트렌디 / w.매출_금액 * 100 END AS `트렌디업종_매출_비중(%)`
    FROM with_label w
),

-- 14. 시계열 지표 및 신규 요구 누락 피처 산출[cite: 1, 2]
with_advanced_features AS (
    SELECT
        f.*,
        -- 2030대 소비 비중 YoY 증가폭(%p)[cite: 2]
        f.`2030대_소비_비중` - LAG(f.`2030대_소비_비중`, 4) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번) AS `2030대_소비_비중_증가(%p, YoY)`,

        -- 최근 4분기간 2030대 소비 비중의 추세기울기[cite: 2]
        (f.`2030대_소비_비중` - LAG(f.`2030대_소비_비중`, 3) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번)) / 3.0 AS `2030비중_추세기울기`,

        -- 유동인구 YoY 증가율 (%)[cite: 2]
        CASE 
            WHEN f.전년동기_유동인구 IS NULL OR f.전년동기_유동인구 = 0 THEN NULL 
            ELSE (f.분기별_총_유동인구_수 - f.전년동기_유동인구) / f.전년동기_유동인구 * 100 
        END AS `유동인구_YoY증가율(%)`,

        -- 주말 매출 비중 변화폭 (2분기 전 대비)[cite: 2]
        f.주말_매출_비중 - LAG(f.주말_매출_비중, 2) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번) AS `주말비중_변화폭_2분기`,

        -- 전환율 %p 변화 (전년 동기 대비)
        f.구매전환율_100cap - LAG(f.구매전환율_100cap, 4) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번) AS `전환율_%p변화`,

        -- 유동인구 연속 개선 분기수 산출
        (CASE WHEN f.분기별_총_유동인구_수 > LAG(f.분기별_총_유동인구_수, 1) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번) THEN 1 ELSE 0 END +
         CASE WHEN LAG(f.분기별_총_유동인구_수, 1) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번) > LAG(f.분기별_총_유동인구_수, 2) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번) THEN 1 ELSE 0 END +
         CASE WHEN LAG(f.분기별_총_유동인구_수, 2) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번) > LAG(f.분기별_총_유동인구_수, 3) OVER (PARTITION BY f.상권_코드 ORDER BY f.분기순번) THEN 1 ELSE 0 END
        ) AS `유동인구_연속개선_분기수`
    FROM with_features f
),

-- 15. 핵심 조건(유동인구_YoY>0 & 전환율_%p변화>0) 동시 충족 여부 플래그
with_condition_flag AS (
    SELECT
        af.*,
        af.`유동인구_YoY증가율(%)` AS `유동인구_YoY(%)`,
        CASE WHEN af.`유동인구_YoY증가율(%)` > 0 AND af.`전환율_%p변화` > 0 THEN 1 ELSE 0 END AS 조건_충족_여부
    FROM with_advanced_features af
),

-- 16. 조건 동시 충족 이력 및 연속성 지표 계산
with_condition_metrics AS (
    SELECT
        cf.*,
        -- 직전 분기(t-1) 조건 충족 여부
        LAG(cf.조건_충족_여부, 1) OVER (PARTITION BY cf.상권_코드 ORDER BY cf.분기순번) AS `조건_동시충족_직전분기`,
        -- 최근 4분기 내 조건 충족 횟수
        SUM(cf.조건_충족_여부) OVER (
            PARTITION BY cf.상권_코드 ORDER BY cf.분기순번
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS `조건_동시충족_최근4분기_충족횟수`
    FROM with_condition_flag cf
),

-- 17. 모델 학습용 최종 피처셋 셀렉션
final AS (
    SELECT
        w.상권_코드,
        w.상권_코드_명,
        w.기준_년분기_코드,
        w.분기순번,
        w.활성화_현재상태,
        LEAD(w.활성화_현재상태, 4) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번) AS 활성화_라벨_1년후,
        -- 1년 후 라벨이 관측된 실제 기준_년분기_코드
        LEAD(w.기준_년분기_코드, 4) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번) AS 라벨_분기_코드,
        -- 학습/평가/실전예측 구간 구분
        CASE
            WHEN w.분기순번 BETWEEN 8 AND 13 THEN 'train'
            WHEN w.분기순번 BETWEEN 14 AND 17 THEN 'test'
            WHEN w.분기순번 BETWEEN 18 AND 21 THEN 'apply_최종예측대상'
        END AS data_split,
        w.매출_YoY_윈저화 AS `매출_YoY_윈저화(%)`,
        w.매출_저점대비_반등폭,
        w.매출_모멘텀,
        w.분기별_총_유동인구_수,
        w.주말_매출_비중 AS `주말_매출_비중(%)`,
        w.`2030대_소비_비중`,
        w.구매전환율_100cap AS `구매전환율_100cap(%)`,
        w.저녁심야_매출_비중 AS `저녁심야_매출_비중(%)`,
        
        -- 추가된 누락 피처 전체 포함[cite: 2]
        w.`2030대_소비_비중_증가(%p, YoY)`,
        w.`2030비중_추세기울기`,
        w.`트렌디업종_매출_비중(%)`,
        w.`유동인구_YoY증가율(%)`,
        w.`유동인구_연속개선_분기수`,
        w.`주말비중_변화폭_2분기`,
        w.`유동인구_YoY(%)`,
        w.`전환율_%p변화`,
        w.`조건_동시충족_직전분기`,
        w.`조건_동시충족_최근4분기_충족횟수`
    FROM with_condition_metrics w
)

-- 18. 최종 대상 필터링 및 출력
SELECT *
FROM final
ORDER BY 상권_코드, 분기순번;
