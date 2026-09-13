# Example data

These files are synthetic and demonstrate the public input interface. They do not contain
Nierji operating records.

| File | Required fields | Time scale |
|---|---|---|
| `reservoir_example.csv` | date, inflow, release, storage, reservoir level, tailwater level | daily |
| `hva_example.csv` | reservoir level, storage, water-surface area | curve |
| `meteorology_example.csv` | timestamp, GHI, DNI, DHI, air temperature, wind speed, grid demand | hourly |
| `grid_example.csv` | generator, capacity, dispatch cost, emission factor | generator |

Units are included in every field name. Replace these files through a separate YAML
configuration; do not add restricted operating data to Git.
