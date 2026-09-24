# Retrieval report

- Backend: `bm25`

## Eval before/after

### Tuning queries

| kind | recall@5 | recall@10 | MRR | n |
|---|---|---|---|---|
| overall | 0.800 → 0.850 | 0.850 → 0.900 | 0.660 → 0.700 | 40 |
| concept | – → 0.700 | – → 0.700 | – → 0.500 | 5 |
| howto | 0.900 → 0.900 | 0.950 → 1.000 | 0.800 → 0.850 | 10 |

### Held-out queries

| kind | recall@5 | recall@10 | MRR | n |
|---|---|---|---|---|
| overall | 0.700 → 0.750 | 0.800 → 0.800 | 0.600 → 0.650 | 10 |

