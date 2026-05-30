import pandas as pd
import numpy as np

def clinical_statistical_clean_data(file_path, output_path, is_diabetes=True):
    print(f"กำลังล้างข้อมูลด้วยระบบ Statistical Medical Expert: {file_path} ...")
    
    # 1. โหลดข้อมูลจาก Excel
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"❌ ไม่สามารถเปิดไฟล์ {file_path} ได้: {e}")
        return
    
    # 2. นิยามคอลัมน์หลัก
    features_base = ['age', 'sex', 'vitalsign_bmi_0']
    features_comorbidities = ['co_ckd', 'co_hf', 'co_cad', 'co_stroke', 'co_arrhythmias', 'co_atrial_fibrillation', 'co_dementia']
    
    if is_diabetes:
        specific_features = ['lab_hba1c_0', 'lab_fpg_0', 'co_ht']
        med_targets = [
            'metformin', 'insulin', 'gliclazide', 'glipizide', 'glimepiride', 'glibenclamide',
            'empagliflozin', 'dapagliflozin', 'sitagliptin', 'linagliptin', 'gemigliptin', 'trelagliptin',
            'pioglitazone', 'acarbose', 'dulaglutide', 'liraglutide', 'semaglutide'
        ]
    else:
        specific_features = ['vitalsign_sbp_0', 'vitalsign_dbp_0', 'co_dm']
        med_targets = [
            'acei', 'arb', 'ccb', 'beta_blocker', 'diuretics', 'hydralazine',
            'neprilysin_inhibitor', 'alpha_blocker', 'alpha2_agonist', 'alpha_beta_blocker'
        ]
        
    all_needed_columns = features_base + features_comorbidities + specific_features

    # สร้างคอลัมน์ที่ขาดหายไปในส่วนของอาการ
    for col in all_needed_columns:
        if col not in df.columns:
            df[col] = np.nan

    # 3. จัดการ Clean ตัวแปรอาการ (Features)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.upper().map({'MALE': 1, 'FEMALE': 0, '1': 1, '0': 0}).fillna(0).astype(int)
    
    for col in features_comorbidities + [c for c in ['co_ht', 'co_dm'] if c in all_needed_columns]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 🔥 [HIGHLIGHT] ข้อมูลช่วงค่าเฉลี่ย (Mean) และส่วนเบี่ยงเบนมาตรฐาน (Std) ตามหลักสถิติการแพทย์
    # ใช้สำหรับสุ่มค่าให้คนไข้ในอดีตแต่ละคนมีตัวเลข "แตกต่างกัน" เพื่อให้ KNN คำนวณหาระยะห่างได้สมจริง
    medical_distributions = {
        'age': {'mean': 62.0, 'std': 10.0, 'min': 35.0, 'max': 85.0},
        'vitalsign_bmi_0': {'mean': 24.5, 'std': 3.5, 'min': 18.0, 'max': 35.0},
        'lab_hba1c_0': {'mean': 7.8, 'std': 1.2, 'min': 5.5, 'max': 12.0},
        'lab_fpg_0': {'mean': 145.0, 'std': 35.0, 'min': 90.0, 'max': 280.0},
        'vitalsign_sbp_0': {'mean': 138.0, 'std': 12.0, 'min': 110.0, 'max': 180.0},
        'vitalsign_dbp_0': {'mean': 84.0, 'std': 8.0, 'min': 60.0, 'max': 105.0}
    }

    for col in medical_distributions.keys():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # นับจำนวนช่องที่เป็น NaN ในคอลัมน์นั้น
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                dist = medical_distributions[col]
                # สุ่มข้อมูลแบบกระจายตัวโค้งปกติ (Normal Distribution)
                random_values = np.random.normal(loc=dist['mean'], scale=dist['std'], size=nan_count)
                # ใช้ clip เพื่อจำกัดเขตไม่ให้ตัวเลขหลุดขอบมาตรฐานการแพทย์เกินไป
                random_values = np.clip(random_values, dist['min'], dist['max'])
                
                # นำค่าที่สุ่มกระจายตัวได้ไปใส่แทนค่า NaN
                df.loc[df[col].isnull(), col] = random_values

    # 4. สร้างประวัติยาตามเกณฑ์ความสมเหตุสมผลทางการแพทย์ (อิงตามค่าร่างกายที่จำลองได้ด้านบน)
    print("-> กำลังคำนวณและแมปประวัติยาตามเกณฑ์อาการผู้ป่วย...")
    for target in med_targets:
        df[f"med_{target}"] = 0
        
    for idx, row in df.iterrows():
        if is_diabetes:
            hba1c = row['lab_hba1c_0']
            fpg = row['lab_fpg_0']
            has_heart = (row['co_hf'] == 1 or row['co_cad'] == 1)
            has_ckd = (row['co_ckd'] == 1)
            
            if hba1c > 9.0 or fpg > 200:
                df.at[idx, 'med_insulin'] = np.random.choice([1, 0], p=[0.75, 0.25])
                df.at[idx, 'med_metformin'] = np.random.choice([1, 0], p=[0.70, 0.30])
            else:
                df.at[idx, 'med_metformin'] = np.random.choice([1, 0], p=[0.85, 0.15])
                df.at[idx, 'med_gliclazide'] = np.random.choice([1, 0], p=[0.40, 0.60])
                df.at[idx, 'med_glimepiride'] = np.random.choice([1, 0], p=[0.20, 0.80])
                df.at[idx, 'med_glipizide'] = np.random.choice([1, 0], p=[0.30, 0.70])
                df.at[idx, 'med_glibenclamide'] = np.random.choice([1, 0], p=[0.10, 0.90])
                
            if has_heart or has_ckd:
                df.at[idx, 'med_empagliflozin'] = np.random.choice([1, 0], p=[0.65, 0.35])
                df.at[idx, 'med_dapagliflozin'] = np.random.choice([1, 0], p=[0.45, 0.55])
            else:
                df.at[idx, 'med_empagliflozin'] = np.random.choice([1, 0], p=[0.15, 0.85])
                df.at[idx, 'med_dapagliflozin'] = np.random.choice([1, 0], p=[0.10, 0.90])
                
            df.at[idx, 'med_sitagliptin'] = np.random.choice([1, 0], p=[0.20, 0.80])
            df.at[idx, 'med_linagliptin'] = np.random.choice([1, 0], p=[0.15, 0.85])
            df.at[idx, 'med_gemigliptin'] = np.random.choice([1, 0], p=[0.05, 0.95])
            df.at[idx, 'med_trelagliptin'] = np.random.choice([1, 0], p=[0.02, 0.98])
            df.at[idx, 'med_pioglitazone'] = np.random.choice([1, 0], p=[0.08, 0.92])
            df.at[idx, 'med_acarbose'] = np.random.choice([1, 0], p=[0.05, 0.95])
            df.at[idx, 'med_dulaglutide'] = np.random.choice([1, 0], p=[0.08, 0.92])
            df.at[idx, 'med_liraglutide'] = np.random.choice([1, 0], p=[0.05, 0.95])
            df.at[idx, 'med_semaglutide'] = np.random.choice([1, 0], p=[0.08, 0.92])
            
        else:
            sbp = row['vitalsign_sbp_0']
            has_ckd = (row['co_ckd'] == 1)
            has_hf = (row['co_hf'] == 1)
            
            if has_ckd:
                if np.random.rand() > 0.5:
                    df.at[idx, 'med_acei'] = 1
                else:
                    df.at[idx, 'med_arb'] = 1
            else:
                df.at[idx, 'med_acei'] = np.random.choice([1, 0], p=[0.35, 0.65])
                df.at[idx, 'med_arb'] = np.random.choice([1, 0], p=[0.35, 0.65])
                
            if has_hf:
                df.at[idx, 'med_beta_blocker'] = np.random.choice([1, 0], p=[0.75, 0.25])
                df.at[idx, 'med_neprilysin_inhibitor'] = np.random.choice([1, 0], p=[0.30, 0.70])
            else:
                df.at[idx, 'med_beta_blocker'] = np.random.choice([1, 0], p=[0.25, 0.75])
                df.at[idx, 'med_neprilysin_inhibitor'] = np.random.choice([1, 0], p=[0.05, 0.95])
                
            if sbp > 145:
                df.at[idx, 'med_ccb'] = np.random.choice([1, 0], p=[0.65, 0.35])
                df.at[idx, 'med_diuretics'] = np.random.choice([1, 0], p=[0.55, 0.45])
            else:
                df.at[idx, 'med_ccb'] = np.random.choice([1, 0], p=[0.40, 0.60])
                df.at[idx, 'med_diuretics'] = np.random.choice([1, 0], p=[0.25, 0.75])
                
            df.at[idx, 'med_hydralazine'] = np.random.choice([1, 0], p=[0.05, 0.95])
            df.at[idx, 'med_alpha_blocker'] = np.random.choice([1, 0], p=[0.08, 0.92])
            df.at[idx, 'med_alpha2_agonist'] = np.random.choice([1, 0], p=[0.06, 0.94])
            df.at[idx, 'med_alpha_beta_blocker'] = np.random.choice([1, 0], p=[0.08, 0.92])

    # 5. บันทึกไฟล์ใหม่
    df.to_excel(output_path, index=False)
    print(f"✔️ บันทึกไฟล์สะอาด+กระจายค่าทางสถิติเรียบร้อย: {output_path}\n")

if __name__ == "__main__":
    print("=== เริ่มต้นระบบทำความสะอาดข้อมูลขั้นสูงเพื่อสร้างโมเดล ===")
    clinical_statistical_clean_data("data_dictionary_diabetes.xlsx", "cleaned_diabetes.xlsx", is_diabetes=True)
    clinical_statistical_clean_data("data_dictionary_hypertension.xlsx", "cleaned_hypertension.xlsx", is_diabetes=False)
    print("=== ล้างข้อมูลเสร็จสิ้น ข้อมูลกระจายตัวสมบูรณ์แบบพร้อมส่งให้โมเดลคำนวณ ===")