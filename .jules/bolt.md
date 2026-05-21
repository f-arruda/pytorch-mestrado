## 2024-05-19 - [O(N) Scans in Nested Pandas Loops]
**Learning:** Performing boolean mask indexing (`df[(df['Hour'] == h) & (df['Modelo'] == m)]`) inside a double loop creates massive overhead due to repeated full dataframe scans for each combination of Hour/Model.
**Action:** Use `.groupby()` computed outside the loops and access elements via `get_group()` to reduce lookup complexity from O(N) to roughly O(1) per combination, achieving significantly faster execution (e.g. 7x speedup in benchmarking). Use `try...except KeyError` alongside `df.iloc[:0].copy()` to elegantly handle missing groups while maintaining identical data schema.

## 2024-05-21 - [O(N) Scans in Linke Turbidity Optimization Loop]
**Learning:** Performing boolean mask indexing (`mask_day = (df.index.date == day) & (df['elevation'] > 10) & (df['ghi'] > 10)`) inside a `for day in unique_days:` loop causes massive overhead due to repeated full dataframe scans for each day.
**Action:** Pre-calculate the subset dataframe and group it outside the loop: `df_valid = df[(df['elevation'] > 10) & (df['ghi'] > 10)]` and `grouped_valid = df_valid.groupby(df_valid.index.date)`. Inside the loop, use `grouped_valid.get_group(day)` in a `try...except KeyError` block to reduce lookup complexity from O(N) to roughly O(1) per combination. This achieves roughly 100x speedup in the preprocessing pipeline for this function.
