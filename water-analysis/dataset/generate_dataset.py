import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000

# WHO guidelines for drinking water quality
ph          = np.random.normal(7.0, 0.8, n).clip(0, 14)
hardness    = np.random.normal(200, 80, n).clip(0, 500)
solids      = np.random.normal(22000, 8000, n).clip(0, 60000)
chloramines = np.random.normal(7.0, 1.5, n).clip(0, 15)
sulfate     = np.random.normal(333, 40, n).clip(0, 500)
conductivity= np.random.normal(425, 80, n).clip(0, 800)
organic_carbon = np.random.normal(14, 3, n).clip(0, 30)
trihalomethanes= np.random.normal(66, 16, n).clip(0, 120)
turbidity   = np.random.normal(4.0, 1.5, n).clip(0, 10)

# WQI calculation (weighted, 0-100, higher = better quality)
ph_score  = 100 - np.abs(ph - 7.0) * 20
hard_score= 100 - np.clip((hardness - 150) / 200 * 100, 0, 100)
solid_score=100 - np.clip(solids / 600, 0, 100)
chlr_score= 100 - np.clip((chloramines - 4) / 8 * 100, 0, 100)
sulf_score= 100 - np.clip((sulfate - 250) / 250 * 100, 0, 100)
cond_score= 100 - np.clip((conductivity - 300) / 500 * 100, 0, 100)
oc_score  = 100 - np.clip(organic_carbon / 30 * 100, 0, 100)
thm_score = 100 - np.clip(trihalomethanes / 120 * 100, 0, 100)
turb_score= 100 - np.clip(turbidity / 10 * 100, 0, 100)

wqi = (
    ph_score   * 0.18 +
    hard_score * 0.10 +
    solid_score* 0.12 +
    chlr_score * 0.12 +
    sulf_score * 0.10 +
    cond_score * 0.08 +
    oc_score   * 0.10 +
    thm_score  * 0.12 +
    turb_score * 0.08
).clip(0, 100)

wqi += np.random.normal(0, 2, n)
wqi = wqi.clip(0, 100)

df = pd.DataFrame({
    'ph': ph, 'Hardness': hardness, 'Solids': solids,
    'Chloramines': chloramines, 'Sulfate': sulfate,
    'Conductivity': conductivity, 'Organic_carbon': organic_carbon,
    'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity,
    'WQI': wqi
})

df.to_csv('/home/claude/water-analysis/dataset/water_quality.csv', index=False)
print(f"Dataset saved: {df.shape}")
print(df.describe().round(2))
