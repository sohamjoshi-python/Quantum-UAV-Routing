## Embedding overhead on Pegasus (embeddable sizes only)

| Requests | Encoding | Logical | Physical qubits | Phys/Logical | Max chain | Mean chain |
|---:|:--|---:|---:|---:|---:|---:|
| 5 | merge_tree | 35 | 42 ± 1 | 1.20× | 2.0 | 1.20 |
| 5 | pairwise | 15 | 25 ± 1 | 1.68× | 2.4 ± 0.5 | 1.68 |
| 10 | merge_tree | 145 | 250 ± 14 | 1.72× | 3.9 ± 0.6 | 1.72 |
| 10 | pairwise | 55 | 259 ± 13 | 4.71× | 7.9 ± 0.6 | 4.71 |
| 20 | merge_tree | 590 | 1,918 ± 121 | 3.25× | 14.6 ± 1.3 | 3.25 |
| 20 | pairwise | 210 | 3,525 ± 270 | 16.79× | 32.3 ± 3.0 | 16.79 |

### Pairwise / merge-tree overhead ratio (both embedded)

| Requests | Physical-qubit ratio | Max-chain ratio |
|---:|---:|---:|
| 5.0 | 0.60× | 1.20× |
| 10.0 | 1.04× | 2.03× |
| 20.0 | 1.84× | 2.21× |