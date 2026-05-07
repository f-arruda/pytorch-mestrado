## 2024-05-19 - [O(N) Scans in Nested Pandas Loops]
**Learning:** Performing boolean mask indexing (`df[(df['Hour'] == h) & (df['Modelo'] == m)]`) inside a double loop creates massive overhead due to repeated full dataframe scans for each combination of Hour/Model.
**Action:** Use `.groupby()` computed outside the loops and access elements via `get_group()` to reduce lookup complexity from O(N) to roughly O(1) per combination, achieving significantly faster execution (e.g. 7x speedup in benchmarking). Use `try...except KeyError` alongside `df.iloc[:0].copy()` to elegantly handle missing groups while maintaining identical data schema.
## 2025-02-23 - [Caching Categorical Identifiers to Avoid O(N) Scans]
**Learning:** Calling `.unique()` on a Pandas DataFrame column (e.g. `df['Modelo'].unique()`) inside loops or frequently called methods causes a full O(N) array scan every time, accumulating significant overhead.
**Action:** Extract and cache unique categorical identifiers as instance variables (e.g., in the `__init__` constructor) to avoid redundant full-column scans during iterative processing or visualization.
