## 2024-05-19 - [O(N) Scans in Nested Pandas Loops]
**Learning:** Performing boolean mask indexing (`df[(df['Hour'] == h) & (df['Modelo'] == m)]`) inside a double loop creates massive overhead due to repeated full dataframe scans for each combination of Hour/Model.
**Action:** Use `.groupby()` computed outside the loops and access elements via `get_group()` to reduce lookup complexity from O(N) to roughly O(1) per combination, achieving significantly faster execution (e.g. 7x speedup in benchmarking). Use `try...except KeyError` alongside `df.iloc[:0].copy()` to elegantly handle missing groups while maintaining identical data schema.
## 2024-05-19 - [O(N) Scans for Chronological Data]
**Learning:** Operations propagating state sequentially, like Linke Turbidity calculated for sequential days, heavily suffer from looping over unique time components and filtering the DataFrame per iteration (`df.loc[df.index.date == day]`).
**Action:** Before looping over a time dimension (like date) sequentially, pre-calculate `df.groupby(df.index.date)` and retrieve each day's chunk with `get_group(day)`, keeping O(1) loop execution while maintaining the necessary state propagation logic.
