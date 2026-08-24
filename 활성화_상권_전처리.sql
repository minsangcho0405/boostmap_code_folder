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
    AND 상권_구분_코드_명 IN ('골목상권', '발달상권')
),

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

with_seq AS (
    SELECT
        dq.*,
        (FLOOR(기준_년분기_코드 / 10) - 2021) * 4
            + MOD(기준_년분기_코드, 10) AS 분기순번
    FROM district_quarter dq
),

with_footfall AS (
    SELECT
        w.*,
        f.총_유동인구_수 AS 분기별_총_유동인구_수
    FROM with_seq w
    LEFT JOIN raw_footfall f
        ON w.상권_코드 = f.상권_코드
       AND w.기준_년분기_코드 = f.기준_년분기_코드
),

with_yoy AS (
    SELECT
        w.*,
        LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) AS 전년동기매출,
        CASE
            WHEN LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) IS NULL THEN NULL
            WHEN LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번) < 5000000 THEN NULL
            ELSE (매출_금액 - LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번))
                 / NULLIF(LAG(매출_금액, 4) OVER (PARTITION BY 상권_코드 ORDER BY 분기순번), 0) * 100
        END AS 매출_YoY_원본
    FROM with_footfall w
),

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
        CASE WHEN w.매출_금액 = 0 THEN NULL ELSE w.매출_2030 / w.매출_금액 END AS `2030대_소비_비중`
    FROM with_label w
),

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
