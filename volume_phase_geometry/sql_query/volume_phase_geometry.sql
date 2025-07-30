WITH base_data AS (
  SELECT 
    timestamp, 
    symbol,
    trade_id, 
    price, 
    quantity, 
    LAG(quantity) OVER (
      ORDER BY 
        timestamp, 
        trade_id
    ) AS prev_quantity, 
    LAG(timestamp) OVER (
      ORDER BY 
        timestamp, 
        trade_id
    ) AS prev_timestamp 
  FROM 
    binance_trades 
  WHERE 
    symbol = 'XRPUSDT' 
  ORDER BY 
    timestamp, 
    trade_id
), 
with_delta AS (
  SELECT 
    timestamp, 
    symbol,
    trade_id, 
    price, 
    quantity, 
    prev_quantity, 
    timestamp - prev_timestamp AS delta_tau_raw 
  FROM 
    base_data
), 
with_valid_deltas AS (
  SELECT 
    timestamp, 
    symbol,
    trade_id, 
    price, 
    quantity, 
    prev_quantity, 
    delta_tau_raw, 
    CASE WHEN delta_tau_raw > 0 THEN delta_tau_raw ELSE NULL END AS valid_delta 
  FROM 
    with_delta
), 
with_safe_delta AS (
  SELECT 
    timestamp, 
    symbol,
    trade_id, 
    price, 
    quantity, 
    prev_quantity, 
    delta_tau_raw, 
    LAST_VALUE(valid_delta) IGNORE NULLS OVER (
      ORDER BY 
        timestamp, 
        trade_id ROWS BETWEEN UNBOUNDED PRECEDING 
        AND CURRENT ROW
    ) AS associated_delta 
  FROM 
    with_valid_deltas
), 
with_frequency AS (
  SELECT 
    timestamp,
    symbol, 
    trade_id, 
    price, 
    quantity, 
    prev_quantity, 
    delta_tau_raw, 
    associated_delta, 
    CASE WHEN delta_tau_raw = 0 THEN associated_delta ELSE delta_tau_raw END AS delta_tau_safe, 
    CASE WHEN delta_tau_raw = 0 THEN 'SINGULARITY_FIXED' WHEN delta_tau_raw IS NULL THEN 'FIRST_TRADE' ELSE 'NORMAL' END AS status, 
    CASE WHEN prev_quantity > 0 
    AND (
      CASE WHEN delta_tau_raw = 0 THEN associated_delta ELSE delta_tau_raw END
    ) > 0 THEN 2 * PI() * (quantity - prev_quantity) / prev_quantity / (
      CASE WHEN delta_tau_raw = 0 THEN associated_delta ELSE delta_tau_raw END
    ) ELSE NULL END AS frequency 
  FROM 
    with_safe_delta
), 
with_theta AS (
  SELECT 
    timestamp, 
    symbol,
    trade_id, 
    price, 
    quantity, 
    prev_quantity, 
    delta_tau_raw, 
    associated_delta, 
    delta_tau_safe, 
    status, 
    frequency, 
    SUM(frequency * delta_tau_safe) OVER (
      ORDER BY 
        timestamp, 
        trade_id ROWS UNBOUNDED PRECEDING
    ) AS theta 
  FROM 
    with_frequency 
  WHERE 
    frequency IS NOT NULL
) 
SELECT 
  timestamp, 
  symbol,
  trade_id, 
  price, 
  quantity, 
  prev_quantity, 
  delta_tau_raw, 
  associated_delta, 
  delta_tau_safe, 
  status, 
  frequency, 
  theta, 
  COS(theta) AS cos_theta, 
  SIN(theta) AS sin_theta, 
  SUM(
    COS(theta)
  ) OVER (
    ORDER BY 
      timestamp, 
      trade_id ROWS UNBOUNDED PRECEDING
  ) AS cumulative_cos_theta, 
  SUM(
    SIN(theta)
  ) OVER (
    ORDER BY 
      timestamp, 
      trade_id ROWS UNBOUNDED PRECEDING
  ) AS cumulative_sin_theta 
FROM 
  with_theta 
ORDER BY 
  timestamp, 
  trade_id;