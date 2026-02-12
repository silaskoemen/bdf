# Best Values

| Parameter        | DGP                 |   Best Value |   Best LOG_LOSS |   Default LOG_LOSS |   Improvement (%) |
|:-----------------|:--------------------|-------------:|----------------:|-------------------:|------------------:|
| alpha            | make_classification |        0.5   |          0.3299 |             0.33   |               0   |
| alpha            | moons               |        0.1   |          0.1186 |             0.1186 |               0.1 |
| alpha            | circles             |        0.1   |          0.0638 |             0.0638 |               0.1 |
| gamma            | make_classification |        0.1   |          0.3302 |             0.3302 |               0   |
| gamma            | moons               |        0.001 |          0.1185 |             0.1187 |               0.2 |
| gamma            | circles             |        0     |          0.0638 |             0.0639 |               0.1 |
| delta            | make_classification |        0.01  |          0.3302 |             0.3302 |               0   |
| delta            | moons               |        0     |          0.1186 |             0.1187 |               0   |
| delta            | circles             |        0.1   |          0.0637 |             0.0638 |               0.2 |
| min_samples_leaf | make_classification |        5     |          0.3236 |             0.3302 |               2   |
| min_samples_leaf | moons               |        5     |          0.116  |             0.1186 |               2.2 |
| min_samples_leaf | circles             |        5     |          0.0567 |             0.0638 |              11.1 |
