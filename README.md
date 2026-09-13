# D-CHyFLO

D-CHyFLO (Dynamic Carbon-aware Hydro–Floating Photovoltaic Lifecycle-Oriented
Framework) evaluates how a conventional hydropower reservoir can operate with floating
photovoltaic (FPV) generation under a shared export limit, and how the carbon benefit
changes over 30 years.

[![tests](https://github.com/DOI-SPY/D-CHyFLO/actions/workflows/ci.yml/badge.svg)](https://github.com/DOI-SPY/D-CHyFLO/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## 1. What this project does

- Reconstructs reservoir level, storage balance, net head and hydropower generation.
- Calculates hourly FPV generation and winter availability scenarios.
- Applies the 250 MW shared export limit and reports FPV use, curtailment and congestion.
- Compares four operating cases to separate direct FPV, hydro flexibility and additional
  carbon-timing effects.
- Calculates Marginal Water-Carbon Value (MCV), capacity response and a parameterized
  30-year lifecycle carbon ledger.

Nierji is a conventional reservoir, not a pumped-storage plant. Reservoir operation can
shift hydropower through time, but the model contains no pumping cycle.

## 2. Main questions

1. Under a shared export limit, how much carbon benefit does reservoir flexibility add
   beyond direct FPV generation, and how does that value change with hydrology and winter
   FPV availability?
2. When FPV lifecycle burdens, grid decarbonization and conditional reservoir greenhouse
   gas (GHG) responses are included, what controls the 30-year net-carbon benefit and how
   far are the registered cases from break-even?

## 3. Workflow

```text
Reservoir and weather data
  -> Hydro and FPV generation
  -> Shared export dispatch
  -> CF0-CF3 carbon attribution with full grid redispatch
  -> MCV and capacity analysis
  -> 30-year net-carbon accounting and break-even diagnostics
```

The four counterfactual cases are:

- **CF0:** hydro-only baseline.
- **CF1:** add FPV while keeping baseline hydro timing.
- **CF2:** allow hydro-FPV redispatch under the same physical constraints.
- **CF3:** add carbon-aware timing without reducing retained hydro energy.

Final operational carbon reduction is calculated from the emission difference between
complete grid redispatch runs. Marginal emission factors (MEFs) are used only to interpret
timing, examine local marginal response and diagnose validity. MEF is not used to calculate
the final total operational carbon reduction.

## 4. Key results

The frozen registered results are:

| Result | Registered value |
|---|---:|
| Direct FPV contribution | 94.30% |
| Hydro-flexibility contribution | 5.67% |
| Additional carbon-timing contribution | 0.030% |
| Interaction | 0 |
| MEF actual-shift validity coverage at the 10% threshold | about 93.235% |
| MCV range | about 0–0.0655 kg CO2/m3 |
| Median MCV | about 0.0499 kg CO2/m3 |
| Capacity response transition | about 275–400 MWac, depending on state |
| Registered 30-year paths | 2,736 |
| Positive at Year 30 | 2,736 within the registered domain |
| Model envelope at Year 30 | about 1.32–17.66 Mt CO2e |
| Carbon payback | 1–10 years; median 3 years |

These results describe the registered model domain and are not construction
recommendations or site-certified carbon values.

Most operational carbon benefit comes directly from FPV generation. Reservoir flexibility
adds a smaller measurable benefit by moving hydropower away from hours when FPV needs the
shared export channel. Further hourly carbon-aware timing adds only a very small benefit.

MCV measures how much additional carbon reduction is obtained from a small change in
reservoir water availability under the model constraints. It is neither a water price nor
a carbon price.

The capacity response transition is where adding more FPV begins to provide noticeably
less carbon benefit per added MW. The 275–400 MWac range changes with hydrology and winter
availability. It is not a construction sizing recommendation. The 500 MWac value is only
the largest candidate that passes the registered geometric screen for at least one tested
DC/AC design.

All registered paths remain positive at Year 30 within the registered parameter domain.
This is a deterministic model result, not a probability. Grid decarbonization is a major
driver of long-term benefit erosion; FPV degradation also reduces the benefit. The
registered ±10% covered-area reservoir GHG response does not change the sign, while joint
stressors reduce the distance to break-even.

## 5. Repository structure

```text
config/          formal defaults and public example settings
data_example/    synthetic input tables and field definitions
src/dchyflo/     calculation modules
scripts/         command-line analysis entry points
figures/         scripts for the key result plots
outputs_example/ output descriptions
tests/           physical, accounting and regression checks
```

Site-specific inputs belong in `data/raw_private/`, which is ignored by Git.

## 6. Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The public example uses a transparent merit-order full-grid redispatch. A private case
study may replace this component with a network-constrained solver through the same
input-output boundary.

## 7. Quick start

```bash
python scripts/run_all.py --config config/example.yaml
pytest
```

The first command runs only synthetic data and writes tables to `outputs/`. Plot the
main results after the run:

```bash
python figures/plot_operational_attribution.py
python figures/plot_mef_validity.py
python figures/plot_mcv_patterns.py
python figures/plot_capacity_response.py
python figures/plot_capacity_knee.py
python figures/plot_lifecycle_trajectory.py
python figures/plot_break_even.py
```

Use `config/default.yaml` as the starting point for private case-study inputs.

## 8. Input data

| Dataset | Description | Required fields | Time scale | Public |
|---|---|---|---|---|
| Reservoir | Flow, storage and operating state | date, inflow, release, storage, reservoir level, tailwater level | daily | template only |
| H-V-A | Level-storage-area relation | reservoir level, storage, area | curve | synthetic example |
| Meteorology | NASA POWER-compatible FPV input | timestamp, GHI, DNI, DHI, air temperature, wind speed | hourly | synthetic example |
| Grid | Generator fleet for redispatch | generator, capacity, dispatch cost, emission factor | generator/hour | synthetic example |
| Lifecycle | Burden, degradation and event assumptions | supplied through YAML | annual | public parameters |
| Reservoir GHG | Background reference and covered-area response | supplied through user data/YAML | annual | conditional interface |

Site-specific reservoir operating data are not redistributed in this repository. The
public code uses standard input templates so that users can provide their own reservoir
data. The formal historical reconstruction covers 6,575 daily records from 2007–2024,
but those restricted records are not included.

Hourly FPV uses irradiance, plane-of-array conversion, cell temperature, DC output,
inverter clipping and AC output. The three registered winter availability scenarios are
`no_cold_derate`, `moderate_cold` and `conservative_cold`. They represent operating
pressure from reduced winter output, not structural ice or snow-load simulation.

## 9. Outputs

The core output tables are:

- `operational_attribution.csv`
- `mcv_summary.csv`
- `capacity_summary.csv`
- `lifecycle_summary.csv`
- `break_even_summary.csv`

Supporting outputs include hourly hydro, FPV and counterfactual tables, MEF diagnostics,
the annual lifecycle ledger, conditional reservoir GHG cases and
`frozen_result_check.csv`. The example values are demonstrations; the frozen result
check records the registered case-study values without exposing restricted source data.

The capacity scan covers 150–550 MWac and DC/AC ratios from 1.0–1.5. It reports water
coverage, clipping, curtailment, export congestion, operational avoided carbon and
marginal avoided carbon per added MW. Surface power density is 98.75 MWp/km2 and the
registered coverage screen is 2.8%.

The lifecycle module uses 20, 36 and 66 g CO2e/kWh cases plus 94 g CO2e/kWh as a broad
parametric stress anchor. It records initial burden at Year 0, a registered replacement
event at Year 15 and end-of-life at Year 30, with no default Module D credit. The 32
conditional Re-Emission cases keep reservoir background flux separate from the
FPV-covered-area increment. Their −10%, 0 and +10% responses are stress assumptions, not
measured causal effects at Nierji.

The expanded break-even search identifies conditions where cumulative 30-year net carbon
reaches zero. It estimates how far registered cases are from losing their benefit. Its
expanded values are diagnostics, not probabilities or future predictions.

## 10. Interpretation limits

- The model does not provide a construction-level FPV sizing recommendation.
- Winter availability is an operating stress scenario, not structural ice-load validation.
- Reservoir GHG response is conditional and applies only to the FPV-covered area increment.
- Grid emissions are model-based and should be updated when better regional data are available.

D-CHyFLO supports research and early-stage screening. It does not replace FPV structural
or anchoring design, ice-load analysis, grid-connection engineering, ecological assessment,
construction safety review or site-certified lifecycle assessment.

## 11. Citation

Use [`CITATION.cff`](CITATION.cff) to cite this software.

[This earlier publication](https://doi.org/10.1016/j.ejrh.2026.103746) provides part of the
reservoir research background. It is a separate, advisor-led study. D-CHyFLO is a new
study with different research questions, methods and outputs.
