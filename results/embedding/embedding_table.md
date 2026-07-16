## Embedding overhead on Pegasus (embeddable sizes only)

| Requests | Encoding | Logical | Physical qubits | Phys/Logical | Max chain | Mean chain |
|---:|:--|---:|---:|---:|---:|---:|
| 5 | merge_tree | 35 | 44 ± 2 | 1.25× | 2.0 | 1.25 |
| 5 | pairwise | 15 | 25 ± 2 | 1.68× | 2.4 ± 0.5 | 1.68 |
| 10 | merge_tree | 145 | 248 ± 14 | 1.72× | 3.9 ± 0.6 | 1.72 |
| 10 | pairwise | 55 | 255 ± 8 | 4.65× | 7.7 ± 0.5 | 4.65 |
| 20 | merge_tree | 590 | 1,941 ± 108 | 3.29× | 15.2 ± 1.5 | 3.29 |
| 20 | pairwise | 210 | 3,434 ± 156 | 16.35× | 31.1 ± 2.5 | 16.35 |

### Pairwise / merge-tree overhead ratio (both embedded)

| Requests | Physical-qubit ratio | Max-chain ratio |
|---:|---:|---:|
| 5.0 | 0.58× | 1.20× |
| 10.0 | 1.03× | 1.97× |
| 20.0 | 1.77× | 2.05× |