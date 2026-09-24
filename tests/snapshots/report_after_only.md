# Retrieval report

- Backend: `bm25`

## Eval before/after

### Tuning queries

| kind | recall@5 | recall@10 | MRR | nDCG@5 | nDCG@10 | n |
|---|---|---|---|---|---|---|
| overall | – → 0.850 | – → 0.900 | – → 0.700 | – → 0.680 | – → 0.720 | 40 |
| concept | – → 0.700 | – → 0.700 | – → 0.500 | – → 0.480 | – → 0.520 | 5 |
| howto | – → 0.900 | – → 1.000 | – → 0.850 | – → 0.830 | – → 0.870 | 10 |

### Held-out queries

| kind | recall@5 | recall@10 | MRR | nDCG@5 | nDCG@10 | n |
|---|---|---|---|---|---|---|
| overall | – → 0.750 | – → 0.800 | – → 0.650 | – → 0.630 | – → 0.670 | 10 |

