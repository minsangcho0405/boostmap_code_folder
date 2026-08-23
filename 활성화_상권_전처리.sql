-- ============================================================
-- 상권 활성화 예측모델 - 전처리 파이프라인 (MySQL 8.0+)
-- 원본: 서울시 상권분석서비스 추정매출-상권 (raw_sales)
-- 참고 별도 소스: 서울시 상권분석서비스 길단위인구 (raw_footfall)
-- ============================================================

-- ------------------------------------------------------------
-- STEP 0. 외식업 10종 필터링
--   439,141행 → 135,981행 (원본 문서 기준)
-- ------------------------------------------------------------
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
        주말_매출_금액
    FROM raw_sales
    WHERE 서비스_업종_코드_명 IN (
        '한식음식점','중식음식점','일식음식점','양식음식점','분식전문점',
        '패스트푸드점','호프-간이주점','치킨전문점','커피-음료','제과점'
    )
),

-- ------------------------------------------------------------
-- STEP 1. 상권 단위 집계 (상권_코드 + 기준_년분기_코드)
--   29,615개 상권-분기 조합
-- ------------------------------------------------------------
district_quarter AS (
    SELECT
        상권_코드,
        MAX(상권_코드_명) AS 상권_코드_명,
        기준_년분기_코드,
        SUM(당월_매출_금액) AS 매출_금액,
        SUM(당월_매출_건수) AS 매출_건수,
        SUM(연령대_20_매출_금액 + 연령대_30_매출_금액) AS 매출_2030,
        SUM(`시간대_17~21_매출_금액` + `시간대_21~24_매출_금액`) AS 매출_저녁심야,
        SUM(주말_매출_금액) AS 매출_주말
    FROM filtered_raw
    GROUP BY 상권_코드, 기준_년분기_코드
),

-- ------------------------------------------------------------
-- STEP 2. 분기 코드 정수화 (기준_년분기_코드 → 분기순번)
--   기준_년분기_코드 형식: YYYYQ (예: 20211 = 2021년 1분기)
--   분기순번 1 = 2021Q1 로 고정
-- ------------------------------------------------------------
with_seq AS (
    SELECT
        dq.*,
        (FLOOR(기준_년분기_코드 / 10) - 2021) * 4
            + MOD(기준_년분기_코드, 10) AS 분기순번
    FROM district_quarter dq
),

-- ------------------------------------------------------------
-- STEP 3. 유동인구 데이터 병합 (별도 소스, 상권_코드+기준_년분기_코드 조인)
-- ------------------------------------------------------------
with_footfall AS (
    SELECT
        w.*,
        f.총_유동인구_수 AS 분기별_총_유동인구_수
    FROM with_seq w
    LEFT JOIN raw_footfall f
        ON w.상권_코드 = f.상권_코드
       AND w.기준_년분기_코드 = f.기준_년분기_코드
),

-- ------------------------------------------------------------
-- STEP 4. 전년동기(YoY) 매출 계산 + 이상치 필터(500만원 미만 제외)
--   전년동기 = 4분기 전(LAG 4) 매출
-- ------------------------------------------------------------
with_yoy AS (
    SELECT
        w.*,
        LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) AS 전년동기매출,
        CASE
            WHEN LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) IS NULL THEN NULL
            WHEN LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) < 5000000 THEN NULL
            ELSE (매출_금액 - LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번))
                 / LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) * 100
        END AS 매출_YoY_원본
    FROM with_footfall w
),

-- ------------------------------------------------------------
-- STEP 5. 윈저화 (분기별 1~99 percentile 클리핑)
--   MySQL은 PERCENTILE_CONT 미지원 → PERCENT_RANK 기반 근사(nearest-rank)로
--   분기별 1%, 99% 지점 값을 구해 서브쿼리로 조인 후 클리핑
-- ------------------------------------------------------------
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

-- ------------------------------------------------------------
-- STEP 6. 최근4분기_매출합 / 직전4분기_매출합 / 누적4분기_성장률
--   최근4분기 = 현재 분기 포함 직전 3개 + 현재(ROWS 3 PRECEDING~CURRENT)
--   직전4분기 = 최근4분기를 4분기 전으로 LAG
-- ------------------------------------------------------------
with_rolling AS (
    SELECT
        w.*,
        SUM(매출_금액) OVER (
            PARTITION BY 상권_코드 ORDER BY 분기순번
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS 최근4분기_매출합
    FROM with_winsorized w
),
with_prev4 AS (
    SELECT
        w.*,
        LAG(최근4분기_매출합, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) AS 직전4분기_매출합
    FROM with_rolling w
),
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

-- ------------------------------------------------------------
-- STEP 7. 활성화 라벨: 최소규모필터(직전4분기합 하위25% 제외) + CAGR(상위20%)
--   동일 분기(기준_년분기_코드) 내 상대평가
-- ------------------------------------------------------------
with_rank AS (
    SELECT
        w.*,
        PERCENT_RANK() OVER (PARTITION BY 기준_년분기_코드 ORDER BY 직전4분기_매출합) AS 규모_pr,
        PERCENT_RANK() OVER (PARTITION BY 기준_년분기_코드 ORDER BY 누적4분기_성장률) AS cagr_pr
    FROM with_cagr w
    WHERE 직전4분기_매출합 IS NOT NULL AND 누적4분기_성장률 IS NOT NULL
),
with_label AS (
    SELECT
        w.*,
        CASE WHEN 규모_pr >= 0.25 THEN 1 ELSE 0 END AS 규모필터_통과여부,
        CASE WHEN 규모_pr >= 0.25 AND cagr_pr >= 0.80 THEN 1 ELSE 0 END AS 활성화_현재상태
    FROM with_rank w
),

-- ------------------------------------------------------------
-- STEP 8. 최종 5개 피처
--   ① 매출_저점대비_반등폭 = 현재YoY - 최근3분기 중 최저YoY (공식 미검증, 42~77%만 일치)
--   ② 매출_모멘텀 = 최근2분기 평균YoY - 그 이전2분기 평균YoY
--   ③ 분기별_총_유동인구_수
--   ④ 주말_매출_비중(%) = 주말매출 / 전체매출 × 100
--   ⑤ `2030대_소비_비중` = 2030매출 / 전체매출 (0~1 fraction)
-- ------------------------------------------------------------
with_features AS (
    SELECT
        w.*,

        (w.매출_YoY_윈저화 - LEAST(
            LAG(w.매출_YoY_윈저화, 1) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번),
            LAG(w.매출_YoY_윈저화, 2) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번),
            LAG(w.매출_YoY_윈저화, 3) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번)
        )) AS 매출_저점대비_반등폭,

        (
            (w.매출_YoY_윈저화 + LAG(w.매출_YoY_윈저화, 1) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번)) / 2
          -
            (LAG(w.매출_YoY_윈저화, 2) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번)
             + LAG(w.매출_YoY_윈저화, 3) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번)) / 2
        ) AS 매출_모멘텀,

        CASE
            WHEN w.분기별_총_유동인구_수 IS NULL OR w.분기별_총_유동인구_수 = 0 THEN NULL
            ELSE LEAST(w.매출_건수 / w.분기별_총_유동인구_수 * 100, 100)
        END AS 구매전환율_100cap,

        CASE WHEN w.매출_금액 = 0 THEN NULL ELSE w.매출_저녁심야 / w.매출_금액 * 100 END AS 저녁심야_매출_비중,

        CASE WHEN w.매출_금액 = 0 THEN NULL ELSE w.매출_주말 / w.매출_금액 * 100 END AS 주말_매출_비중,

        CASE WHEN w.매출_금액 = 0 THEN NULL ELSE w.매출_2030 / w.매출_금액 END AS `2030대_소비_비중`

    FROM with_label w
),

-- ------------------------------------------------------------
-- STEP 9. 1년 후(4분기 후) 활성화 라벨 부여 (LEAD 4)
--   train/test/apply 분할은 이 SQL 결과가 아니라 Python 단에서 분기순번 기준으로 부여함
--   (train: 8~13 / test: 14~17 / apply: 18~21, 2026Q1까지 raw_sales에 적재되어 있어야 apply=21 확보 가능)
-- ------------------------------------------------------------
final AS (
    SELECT
        w.상권_코드,
        w.상권_코드_명,
        w.기준_년분기_코드,
        w.분기순번,
        w.활성화_현재상태,
        LEAD(w.활성화_현재상태, 4) OVER (PARTITION BY w.상권_코드 ORDER BY w.분기순번) AS 활성화_라벨_1년후,
        w.매출_YoY_윈저화 AS `매출_YoY_윈저화(%)`,
        w.매출_저점대비_반등폭,
        w.매출_모멘텀,
        w.분기별_총_유동인구_수,
        w.주말_매출_비중 AS `주말_매출_비중(%)`,
        w.`2030대_소비_비중`,
        w.구매전환율_100cap AS `구매전환율_100cap(%)`,
        w.저녁심야_매출_비중 AS `저녁심야_매출_비중(%)`
    FROM with_features w
)
SELECT *
FROM final
WHERE 활성화_현재상태 = 0
ORDER BY 상권_코드, 분기순번;
