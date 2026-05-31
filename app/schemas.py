from typing import Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    disease_type: str
    age: float
    sex: str
    vitalsign_bmi_0: float
    lab_hba1c_0: float = 0.0
    lab_fpg_0: float = 0.0
    vitalsign_sbp_0: float = 0.0
    vitalsign_dbp_0: float = 0.0
    co_ckd: int = 0
    co_hf: int = 0
    co_cad: int = 0
    co_stroke: int = 0
    co_arrhythmias: int = 0
    co_atrial_fibrillation: int = 0
    co_dementia: int = 0
    current_medications: list = []


class PredictInput(BaseModel):
    # Disease flags
    has_ht: bool = False
    has_dm: bool = False

    # Demographics
    age: float = Field(..., ge=18, le=120)
    sex: str  # "MALE" or "FEMALE"

    # SBP temporal aggregation
    sbp_mean: float = Field(..., ge=90, le=250)
    sbp_std: float = Field(default=0.0, ge=0)
    sbp_max: float = Field(default=0.0, ge=0)

    # DBP temporal aggregation
    dbp_mean: float = Field(..., ge=40, le=150)
    dbp_std: float = Field(default=0.0, ge=0)
    dbp_max: float = Field(default=0.0, ge=0)

    # HbA1c temporal aggregation
    hba1c_mean: Optional[float] = Field(default=None, ge=0)
    hba1c_std:  Optional[float] = Field(default=None, ge=0)
    hba1c_max:  Optional[float] = Field(default=None, ge=0)

    # Additional labs
    bmi_mean:        float          = Field(..., ge=10, le=70)
    fpg_mean:        Optional[float] = Field(default=None, ge=0)
    hemoglobin_mean: Optional[float] = Field(default=None, ge=0)


class RiskResult(BaseModel):
    probability: float
    percentage: float
    risk_level: str  # LOW | MEDIUM | HIGH


class PredictOutput(BaseModel):
    status: str
    disease_group: str
    predictions: dict
    explanation: str
    timestamp: str
