from dataclasses import dataclass, field
from typing import Optional, Sequence

@dataclass(frozen=True)
class BaselineConfig:
    # data
    data_path: str = "data/raw/"

    # output
    out_dir: str = "artifacts/cate/"

    # columns
    id_col: str = "stay_id"
    subject_col: str = "subject_id"
    treatment_col: str = "t_vaso6h"
    outcome_col: str = "y_hosp_mort"

    # Outcome Nice to Print Name
    outcome_nice_name: Optional[str] = "Mortality"

    # split
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # weighting / overlap
    ps_clip: tuple[float, float] = (0.01, 0.99)        # clip e(x) before weights
    weight_trim_quantiles: tuple[float, float] = (0.01, 0.99)  # trim weights

    # cross-fitting
    n_folds: int = 5

    # feature control
    drop_cols: Optional[Sequence[str]] = field(default_factory=list)

    # tau direction
    tau_direction: str = "lte"  # "gte" or "lte"

    # Policy Tau
    policy: str = "tau_gt_0" # ["tau_gt_0", "tau_lt_0", "top_frac_benefit"]
    top_frac:float = 0.2 # Only used if policy="top_frac_benefit", e.g. top 20% most benefit patients


SEPSIS_V2 = BaselineConfig(
    data_path="data/raw/SEPSIS_V2/sepsis_v2.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="treat_steroid",
    outcome_col="y_28d_mort_inhosp",
    drop_cols=["intime", "t0_time","t0_support", "y_hosp_mort", "SO2_bg",# "y_28d_mort_inhosp"
               "enroll_end", "steroid_unparsed_events", "treat_steroid",
               "steroid_total_events", "hc_equiv_mg_0_24h", "lactate",
               "baseline_lookback_hours", "elig_within_hours", "treat_window_hours",
               "stay_id", "subject_id", "hadm_id"], # "glucose"
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/sepsis_v2",
    tau_direction="lte",
    policy="tau_lt_0",
    n_folds=5
)


DIUR_V1 = BaselineConfig(
    data_path="data/raw/DIUR_V1/diur_v1.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="treat_early",
    outcome_col="y_28d_mort_inhosp",
    drop_cols=["intime", "t0_time","sepsis_time", "exposure_end", "treat_early", "y_28d_mort_inhosp",
                "y_hosp_mort", "dischtime", "deathtime", "stay_id", "subject_id", "hadm_id",
                 "glucose", "hemoglobin", "platelets", "first_diur_time"],
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/diur_v1",
    tau_direction="lte",
    policy="tau_gt_0",
    top_frac=0.2,
    n_folds=5
)


AIDS_V1 = BaselineConfig(
    data_path="data/raw/AIDS_V1/aids_v1.parquet",
    id_col="id",
    subject_col="id",
    treatment_col="treat",
    outcome_col="label",
    drop_cols=["time", "trt", "treat", "id"],
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/aids_v1",
    tau_direction="lte",
    policy="tau_lt_0",
    n_folds=5,
    outcome_nice_name="AIDS Progression"
)

AIDS_V1_BIASED =  BaselineConfig(
    outcome_nice_name = "Biased AIDS Progression",
    data_path = "data/raw/AIDS_V1_BIASED/aids_v1_biased.parquet",
    id_col="id",
    subject_col="id",
    treatment_col="treat",
    outcome_col="label",
    drop_cols=["time", "trt", "treat", "id"],
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/aids_v1_biased",
    tau_direction="lte",
    policy="tau_lt_0",
    n_folds=5,
)

