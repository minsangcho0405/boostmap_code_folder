-- 1. 테이블 생성 (DDL)

DROP TABLE IF EXISTS raw_sales;
DROP TABLE IF EXISTS raw_footfall;

CREATE TABLE IF NOT EXISTS raw_sales (
    기준_년분기_코드     INT,
    상권_구분_코드       VARCHAR(10),
    상권_구분_코드_명    VARCHAR(50),
    상권_코드           INT,
    상권_코드_명         VARCHAR(100),
    서비스_업종_코드     VARCHAR(20),
    서비스_업종_코드_명  VARCHAR(100),
    당월_매출_금액       BIGINT,
    당월_매출_건수       BIGINT,
    주중_매출_금액       BIGINT,
    주말_매출_금액       BIGINT,
    월요일_매출_금액     BIGINT,
    화요일_매출_금액     BIGINT,
    수요일_매출_금액     BIGINT,
    목요일_매출_금액     BIGINT,
    금요일_매출_금액     BIGINT,
    토요일_매출_금액     BIGINT,
    일요일_매출_금액     BIGINT,
    `시간대_00~06_매출_금액` BIGINT,
    `시간대_06~11_매출_금액` BIGINT,
    `시간대_11~14_매출_금액` BIGINT,
    `시간대_14~17_매출_금액` BIGINT,
    `시간대_17~21_매출_금액` BIGINT,
    `시간대_21~24_매출_금액` BIGINT,
    남성_매출_금액       BIGINT,
    여성_매출_금액       BIGINT,
    연령대_10_매출_금액   BIGINT,
    연령대_20_매출_금액   BIGINT,
    연령대_30_매출_금액   BIGINT,
    연령대_40_매출_금액   BIGINT,
    연령대_50_매출_금액   BIGINT,
    연령대_60_이상_매출_금액 BIGINT,
    주중_매출_건수       BIGINT,
    주말_매출_건수       BIGINT,
    월요일_매출_건수     BIGINT,
    화요일_매출_건수     BIGINT,
    수요일_매출_건수     BIGINT,
    목요일_매출_건수     BIGINT,
    금요일_매출_건수     BIGINT,
    토요일_매출_건수     BIGINT,
    일요일_매출_건수     BIGINT,
    `시간대_건수~06_매출_건수` BIGINT,
    `시간대_건수~11_매출_건수` BIGINT,
    `시간대_건수~14_매출_건수` BIGINT,
    `시간대_건수~17_매출_건수` BIGINT,
    `시간대_건수~21_매출_건수` BIGINT,
    `시간대_건수~24_매출_건수` BIGINT,
    남성_매출_건수       BIGINT,
    여성_매출_건수       BIGINT,
    연령대_10_매출_건수   BIGINT,
    연령대_20_매출_건수   BIGINT,
    연령대_30_매출_건수   BIGINT,
    연령대_40_매출_건수   BIGINT,
    연령대_50_매출_건수   BIGINT,
    연령대_60_이상_매출_건수 BIGINT,
    UNIQUE KEY uk_sales (상권_코드, 기준_년분기_코드, 서비스_업종_코드)
);

CREATE TABLE IF NOT EXISTS raw_footfall (
    기준_년분기_코드     INT,
    상권_구분_코드       VARCHAR(10),
    상권_구분_코드_명    VARCHAR(50),
    상권_코드           INT,
    상권_코드_명         VARCHAR(100),
    총_유동인구_수       BIGINT,
    남성_유동인구_수     BIGINT,
    여성_유동인구_수     BIGINT,
    연령대_10_유동인구_수 BIGINT,
    연령대_20_유동인구_수 BIGINT,
    연령대_30_유동인구_수 BIGINT,
    연령대_40_유동인구_수 BIGINT,
    연령대_50_유동인구_수 BIGINT,
    연령대_60_이상_유동인구_수 BIGINT,
    `시간대_00_06_유동인구_수` BIGINT,
    `시간대_06_11_유동인구_수` BIGINT,
    `시간대_11_14_유동인구_수` BIGINT,
    `시간대_14_17_유동인구_수` BIGINT,
    `시간대_17_21_유동인구_수` BIGINT,
    `시간대_21_24_유동인구_수` BIGINT,
    월요일_유동인구_수   BIGINT,
    화요일_유동인구_수   BIGINT,
    수요일_유동인구_수   BIGINT,
    목요일_유동인구_수   BIGINT,
    금요일_유동인구_수   BIGINT,
    토요일_유동인구_수   BIGINT,
    일요일_유동인구_수   BIGINT,
    UNIQUE KEY uk_footfall (상권_코드, 기준_년분기_코드)
);

-- 2. CSV 데이터 임포트
LOAD DATA LOCAL INFILE '/Users/mac/Downloads/서울시_상권활성화_데이터셋(최종) 3/서울시_상권분석서비스(추정매출-상권)_2021년.csv'
INTO TABLE raw_sales
CHARACTER SET euckr
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/Users/mac/Downloads/서울시_상권활성화_데이터셋(최종) 3/서울시_상권분석서비스(추정매출-상권)_2022년.csv'
INTO TABLE raw_sales
CHARACTER SET euckr
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/Users/mac/Downloads/서울시_상권활성화_데이터셋(최종) 3/서울시_상권분석서비스(추정매출-상권)_2023년.csv'
INTO TABLE raw_sales
CHARACTER SET euckr
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/Users/mac/Downloads/서울시_상권활성화_데이터셋(최종) 3/서울시 상권분석서비스(추정매출-상권)_2024년.csv'
INTO TABLE raw_sales
CHARACTER SET euckr
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/Users/mac/Downloads/서울시_상권활성화_데이터셋(최종) 3/서울시 상권분석서비스(추정매출-상권)_2025년+2026년_1분기.csv'
INTO TABLE raw_sales
CHARACTER SET euckr
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

LOAD DATA LOCAL INFILE '/Users/mac/Downloads/서울시_상권활성화_데이터셋(최종) 3/서울시 상권분석서비스(길단위인구-상권).csv'
INTO TABLE raw_footfall
CHARACTER SET euckr
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;
