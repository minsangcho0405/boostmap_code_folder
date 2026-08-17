-- 외식업 10종 필터링
WITH filtered_raw AS (
    SELECT
        상권_코드,
        상권_코드_명,
        기준_년분기_코드,
        당월_매출_금액
    FROM raw_sales
    WHERE 서비스_업종_코드_명 IN (
        '한식음식점','중식음식점','일식음식점','양식음식점','분식전문점',
        '패스트푸드점','호프-간이주점','치킨전문점','커피-음료','제과점'
    )
),

-- 상권·분기 단위 매출 집계
district_quarter AS (
    SELECT
        상권_코드,
        MAX(상권_코드_명) AS 상권_코드_명,
        기준_년분기_코드,
        SUM(당월_매출_금액) AS 매출_금액
    FROM filtered_raw
    GROUP BY 상권_코드, 기준_년분기_코드
),

-- 분기순번 정수화
with_seq AS (
    SELECT
        dq.*,
        (FLOOR(기준_년분기_코드 / 10) - 2021) * 4 + MOD(기준_년분기_코드, 10) AS 분기순번
    FROM district_quarter dq
),

-- 전년동기 대비 YoY 계산 + 500만원 미만 이상치 제외
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
    FROM with_seq w
),

-- 윈저화: 분기별 1%/99% 지점 계산
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
-- 윈저화 적용(clip)
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

-- 최근4분기 매출합
with_rolling AS (
    SELECT
        w.*,
        SUM(매출_금액) OVER (
            PARTITION BY 상권_코드 ORDER BY 분기순번
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS 최근4분기_매출합
    FROM with_winsorized w
),
-- 직전4분기 매출합
with_prev4 AS (
    SELECT
        w.*,
        LAG(최근4분기_매출합, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) AS 직전4분기_매출합
    FROM with_rolling w
),
-- 누적4분기 성장률(CAGR) 계산
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

-- 규모필터·CAGR 동일분기 내 순위 산출
with_rank AS (
    SELECT
        w.*,
        PERCENT_RANK() OVER (PARTITION BY 기준_년분기_코드 ORDER BY 직전4분기_매출합) AS 규모_pr,
        PERCENT_RANK() OVER (PARTITION BY 기준_년분기_코드 ORDER BY 누적4분기_성장률) AS cagr_pr
    FROM with_cagr w
    WHERE 직전4분기_매출합 IS NOT NULL AND 누적4분기_성장률 IS NOT NULL
)

-- 규모필터+CAGR 상위20% 충족 시 활성화 라벨 부여
SELECT
    상권_코드,
    상권_코드_명,
    기준_년분기_코드,
    분기순번,
    직전4분기_매출합,
    최근4분기_매출합,
    누적4분기_성장률,
    CASE WHEN 규모_pr >= 0.25 THEN 1 ELSE 0 END AS 규모필터_통과여부,
    CASE WHEN 규모_pr >= 0.25 AND cagr_pr >= 0.80 THEN 1 ELSE 0 END AS 활성화_현재상태
FROM with_rank
ORDER BY 상권_코드, 분기순번;
