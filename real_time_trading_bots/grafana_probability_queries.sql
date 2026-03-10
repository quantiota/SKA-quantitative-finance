  WITH base_data AS (                                                                                                                                                                                        
    SELECT                                                                                                                                                                                                 
      timestamp,                                                                                                                                                                                             
      symbol,     
      trade_id,
      price,
      entropy,
      LAG(price) OVER (
        ORDER BY timestamp, trade_id
      ) AS prev_price,
      LAG(entropy) OVER (
        ORDER BY timestamp, trade_id
      ) AS prev_entropy
    FROM binance_trades
    WHERE
      symbol = 'XRPUSDT'
      AND entropy IS NOT NULL
    ORDER BY timestamp, trade_id
  ),
  with_regime AS (
    SELECT
      timestamp,
      symbol,
      trade_id,
      price,
      entropy,
      prev_entropy,
      CASE
        WHEN price - prev_price > 0 THEN 1
        WHEN price - prev_price < 0 THEN 2
        ELSE 0
      END AS regime
    FROM base_data
    WHERE prev_price IS NOT NULL
  ),
  with_transition AS (
    SELECT
      timestamp,
      symbol,
      trade_id,
      price,
      entropy,
      prev_entropy,
      regime,
      LAG(regime) OVER (
        ORDER BY timestamp, trade_id
      ) AS prev_regime
    FROM with_regime
  ),
  with_probability AS (
    SELECT
      timestamp,
      symbol,
      trade_id,
      price,
      entropy,
      regime,
      prev_regime,
      prev_regime * 3 + regime AS transition_code,
      CASE
        WHEN entropy != 0 AND prev_entropy IS NOT NULL
        THEN EXP(-ABS((entropy - prev_entropy) / entropy))
        ELSE NULL
      END AS P
    FROM with_transition
    WHERE prev_regime IS NOT NULL
  )
  SELECT
    timestamp,
    symbol,
    trade_id,
    price,
    P,
    transition_code,
    CASE transition_code
        WHEN 0 THEN 'neutral→neutral'
        WHEN 1 THEN 'neutral→bull'
        WHEN 2 THEN 'neutral→bear'
        WHEN 3 THEN 'bull→neutral'
        WHEN 4 THEN 'bull→bull'
        WHEN 5 THEN 'bull→bear'
        WHEN 6 THEN 'bear→neutral'
        WHEN 7 THEN 'bear→bull'
        WHEN 8 THEN 'bear→bear'
    END AS transition_name
  FROM with_probability
  ORDER BY timestamp, trade_id;

