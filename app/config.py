# -------------------------------------------------------
# Feature lists
# -------------------------------------------------------
features_base = ['age', 'sex', 'vitalsign_bmi_0']
features_comorbidity = [
    'co_ckd', 'co_hf', 'co_cad', 'co_stroke',
    'co_arrhythmias', 'co_atrial_fibrillation', 'co_dementia',
]

# -------------------------------------------------------
# Medication targets
# -------------------------------------------------------
DM_MEDS = [
    'metformin', 'insulin', 'gliclazide', 'glipizide', 'glimepiride', 'glibenclamide',
    'empagliflozin', 'dapagliflozin', 'sitagliptin', 'linagliptin', 'gemigliptin',
    'trelagliptin', 'pioglitazone', 'acarbose', 'dulaglutide', 'liraglutide', 'semaglutide',
]
HT_MEDS = [
    'acei', 'arb', 'ccb', 'beta_blocker', 'diuretics', 'hydralazine',
    'neprilysin_inhibitor', 'alpha_blocker', 'alpha2_agonist', 'alpha_beta_blocker',
]

# -------------------------------------------------------
# Feature weights — lab values & comorbidities สำคัญกว่า age/BMI
# -------------------------------------------------------
DM_FEATURE_WEIGHTS = {
    'age':                    0.5,
    'sex':                    0.3,
    'vitalsign_bmi_0':        0.7,
    'co_ckd':                 1.8,
    'co_hf':                  1.8,
    'co_cad':                 1.5,
    'co_stroke':              1.2,
    'co_arrhythmias':         1.0,
    'co_atrial_fibrillation': 1.0,
    'co_dementia':            0.8,
    'lab_hba1c_0':            2.5,   # สำคัญที่สุดสำหรับ DM
    'lab_fpg_0':              2.0,
    'co_ht':                  1.2,
}

HT_FEATURE_WEIGHTS = {
    'age':                    0.5,
    'sex':                    0.3,
    'vitalsign_bmi_0':        0.7,
    'co_ckd':                 1.8,
    'co_hf':                  1.8,
    'co_cad':                 1.5,
    'co_stroke':              1.2,
    'co_arrhythmias':         1.0,
    'co_atrial_fibrillation': 1.0,
    'co_dementia':            0.8,
    'vitalsign_sbp_0':        2.5,   # สำคัญที่สุดสำหรับ HT
    'vitalsign_dbp_0':        2.0,
    'co_dm':                  1.2,
}

# -------------------------------------------------------
# Data paths
# -------------------------------------------------------
DM_DATA_PATH = "data/processed/cleaned_diabetes.xlsx"
HT_DATA_PATH = "data/processed/cleaned_hypertension.xlsx"
