WITH ResampledTrades AS (
		SELECT
		DATE_TRUNC('second', timestamp) AS resampled_timestamp,
		symbol,
		AVG(price) AS average_price,
		SUM(quantity) AS total_quantity
		FROM
		binance_trades
		WHERE
		symbol = 'XRPUSDT'
		GROUP BY
		resampled_timestamp,
		symbol
		),
		
		PreviousValueCalc AS (
		SELECT 
		resampled_timestamp AS timestamp,
		symbol,
		average_price AS price,
		total_quantity AS quantity, 
		FIRST_VALUE(total_quantity) OVER (
		PARTITION BY symbol
		ORDER BY resampled_timestamp
		ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
		) AS previous_value
		FROM 
		ResampledTrades
		),
		
		FrequencyCalc AS (
		SELECT 
		timestamp, 
		symbol,
		price,
		quantity, 
		previous_value,
		(2 * PI() * (quantity - previous_value) / previous_value) AS frequency
		FROM 
		PreviousValueCalc
		WHERE
		previous_value IS NOT NULL AND previous_value > 0
		),
		
		ThetaCalc AS (
		SELECT 
		symbol,
		timestamp,
		price, 
		quantity, 
		frequency,
		SUM(frequency) OVER (
		PARTITION BY symbol
		ORDER BY timestamp
		ROWS UNBOUNDED PRECEDING
		) AS theta
		FROM 
		FrequencyCalc
		)
		
		SELECT 
		symbol,
		timestamp,
		price, 
		quantity, 
		frequency,
		theta,
		COS(theta) AS cos_theta,
		SUM(COS(theta)) OVER (
		PARTITION BY symbol
		ORDER BY timestamp
		ROWS UNBOUNDED PRECEDING
		) AS cumulative_cos_theta,
		SIN(theta) AS sin_theta,
		SUM(SIN(theta)) OVER (
		PARTITION BY symbol
		ORDER BY timestamp
		ROWS UNBOUNDED PRECEDING
		) AS cumulative_sin_theta
		FROM 
		ThetaCalc
		ORDER BY 
		timestamp;