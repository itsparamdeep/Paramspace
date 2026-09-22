import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from scipy import stats

url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
oni = pd.read_csv(url, sep=r"\s+")

# keep only the Oct-Nov-Dec reading each year (peak season for our regions)
peak = oni[oni["SEAS"] == "OND"]
nino = peak[["YR", "ANOM"]].rename(columns={"YR": "year", "ANOM": "nino"})

print(nino.tail(10))
print(nino.sort_values("nino", ascending=False).head(8))

# ============================================================
# STEP 2: Load coffee production data (Colombia, Central America, Brazil)
# Why: These are the regions El Nino affects most directly.
# Colombia and Central America dry out under El Nino (clear link).
# Brazil's effect is mixed (rain in south, dryness in north).
# We compare all three against the El Nino numbers from Step 1.
# ============================================================

url = "https://apps.fas.usda.gov/psdonline/downloads/psd_coffee_csv.zip"

# USDA publishes this as a zipped CSV, pandas can read it directly from the zip
coffee = pd.read_csv(url, compression="zip")

# Keep only the countries we care about
countries = ["Colombia", "Honduras", "Guatemala", "Brazil"]
coffee = coffee[coffee["Country_Name"].isin(countries)]

# Keep only the production row (the file has exports, stocks, etc. mixed in)
coffee = coffee[coffee["Attribute_Description"] == "Production"]

# Simplify to just year, country, and value
coffee = coffee[["Country_Name", "Market_Year", "Value"]].rename(
    columns={"Country_Name": "country", "Market_Year": "year", "Value": "production"}
)

print(coffee.head(10))
print(coffee["country"].unique())

# ============================================================
# STEP 3: Merge El Nino readings with the harvest they actually affect
# Why: El Nino peaks Oct-Nov-Dec. That affects the harvest that
# gets counted the FOLLOWING year, not the same year.
# So we shift the nino data forward by one year before merging.
# ============================================================

import pandas as pd

# --- rebuild the two tables from steps 1 and 2 ---
url_nino = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
oni = pd.read_csv(url_nino, sep=r"\s+")
peak = oni[oni["SEAS"] == "OND"]
nino = peak[["YR", "ANOM"]].rename(columns={"YR": "year", "ANOM": "nino"})

url_coffee = "https://apps.fas.usda.gov/psdonline/downloads/psd_coffee_csv.zip"
coffee = pd.read_csv(url_coffee, compression="zip")
countries = ["Colombia", "Honduras", "Guatemala", "Brazil"]
coffee = coffee[coffee["Country_Name"].isin(countries)]
coffee = coffee[coffee["Attribute_Description"] == "Production"]
coffee = coffee[["Country_Name", "Market_Year", "Value"]].rename(
    columns={"Country_Name": "country", "Market_Year": "year", "Value": "production"}
)

# --- the key step: shift nino year forward by 1 ---
# a nino reading in year Y is compared against harvest year Y+1
nino["harvest_year"] = nino["year"] + 1

# --- merge on harvest_year matching coffee's year ---
merged = coffee.merge(nino, left_on="year", right_on="harvest_year", suffixes=("", "_nino"))
merged = merged[["country", "year", "production", "nino"]]

print(merged.head(15))
print(merged.shape)

# --- run one regression per country ---
from sklearn.linear_model import LinearRegression
from scipy import stats

for country in merged["country"].unique():
    sub = merged[merged["country"] == country]

    X = sub[["nino"]].values
    y = sub["production"].values

    model = LinearRegression().fit(X, y)
    slope, intercept, r, p, std_err = stats.linregress(sub["nino"], sub["production"])

    print(f"\n{country}")
    print(f"  slope: {slope:.1f}  (production change per +1 nino strength)")
    print(f"  p-value: {p:.3f}  (below 0.05 means the link is likely real, not noise)")
    print(f"  r-squared: {r**2:.3f}  (how much of the variation nino explains)")

# ============================================================
# STEP 5: Analog years
# Question: in the strongest past El Nino years, what happened
# to production, and roughly when did prices seem to react?
# This is the "human" check on the regression's average number.
# ============================================================

# the strongest El Nino years on record, from step 1's ranking
analog_years = [2015, 1997, 1982, 2023, 1972, 1965]

for y in analog_years:
    harvest_year = y + 1
    sub = merged[merged["year"] == harvest_year]

    if sub.empty:
        print(f"\n{y} -> harvest {harvest_year}: no data")
        continue

    print(f"\nEl Nino year {y}  ->  harvest year {harvest_year}")
    for _, row in sub.iterrows():
        print(f"  {row['country']}: {row['production']:.0f}  (nino was {row['nino']:.2f})")    

 # ============================================================
# STEP 6: Compare today's El Nino reading against the futures spread
# Question: does the Dec26 vs Jul27 spread already reflect
# what the regression + analog years say should happen?
# ============================================================

# --- current El Nino reading ---
# NOAA's latest RONI reading (check cpc.ncep.noaa.gov for the newest number,
# ONI won't have OND finalized until January)
today_nino = 2.0  # placeholder — replace with the latest published RONI value

# --- predicted production change, using Colombia as the clearest case ---
# (swap "Colombia" for whichever country had a real p-value in step 4)
target_country = "Colombia"
sub = merged[merged["country"] == target_country]

from sklearn.linear_model import LinearRegression
X = sub[["nino"]].values
y = sub["production"].values
model = LinearRegression().fit(X, y)

predicted_production = model.predict([[today_nino]])[0]
avg_production = sub["production"].mean()
predicted_change_pct = (predicted_production - avg_production) / avg_production * 100

print(f"{target_country}")
print(f"  today's nino reading: {today_nino}")
print(f"  predicted production: {predicted_production:.0f}")
print(f"  average production:   {avg_production:.0f}")
print(f"  predicted change:     {predicted_change_pct:.1f}%")

# --- you fill these in from your broker or a data feed ---
dec26_price = None   # e.g. 285.50 (cents per lb)
jul27_price = None   # e.g. 292.00

if dec26_price and jul27_price:
    spread = jul27_price - dec26_price
    spread_pct = spread / dec26_price * 100
    print(f"\nDec26 price: {dec26_price}")
    print(f"Jul27 price: {jul27_price}")
    print(f"Spread: {spread:.2f}  ({spread_pct:.1f}%)")
else:
    print("\nFill in dec26_price and jul27_price to compute the spread.")

    # ============================================================
# STEP 7 (revised): Verdict, with specific checkpoints
# Instead of a vague summary, this pulls the actual numbers from
# steps 4-6 and states what each one implies, one at a time.
# ============================================================

print("=" * 55)
print("VERDICT CHECKLIST")
print("=" * 55)

# --- checkpoint 1: is the regression trustworthy at all? ---
print(f"\n[1] Statistical reliability ({target_country})")
print(f"    p-value: {p:.3f}")
if p < 0.05:
    print(f"    -> Real signal. Unlikely to be random chance.")
else:
    print(f"    -> NOT reliable. Treat everything below as speculation, not evidence.")

# --- checkpoint 2: how big is the effect, in real terms? ---
print(f"\n[2] Size of the effect")
print(f"    slope: {slope:.1f} tonnes of production per +1 nino strength")
print(f"    r-squared: {r**2:.3f}")
print(f"    -> El Nino explains {r**2*100:.0f}% of year-to-year production swings.")
print(f"       The other {100-r**2*100:.0f}% is disease, replanting cycles, price incentives, etc.")

# --- checkpoint 3: does history actually agree? ---
print(f"\n[3] Analog year consistency")
print(f"    Look at the step 5 printout: in how many of the 6 strongest")
print(f"    El Nino years did {target_country} production actually fall")
print(f"    versus its neighboring years? If it's 4+ of 6, the regression")
print(f"    isn't just an average masking random noise.")

# --- checkpoint 4: what does today's reading imply ---
print(f"\n[4] Today's predicted shortfall")
print(f"    today's nino reading: {today_nino}")
print(f"    predicted change: {predicted_change_pct:.1f}% vs historical average")

# --- checkpoint 5: is it already priced in ---
print(f"\n[5] Market pricing check")
if dec26_price and jul27_price:
    print(f"    Dec26/Jul27 spread: {spread_pct:.1f}%")
    print(f"    predicted shortfall: {predicted_change_pct:.1f}%")
    gap = abs(predicted_change_pct) - spread_pct
    print(f"    gap: {gap:.1f} percentage points")
    if gap > 2:
        print(f"    -> Spread looks too small for the predicted shortfall.")
        print(f"       Worth investigating further, not a green light to trade.")
    elif gap < -2:
        print(f"    -> Spread looks bigger than the shortfall justifies.")
        print(f"       Market may be pricing in more than the data supports.")
    else:
        print(f"    -> Spread roughly matches the predicted shortfall.")
        print(f"       Looks like the market has already caught up.")
else:
    print(f"    Fill in dec26_price and jul27_price to see this.")


# fixed folder, created once, reused every time
data_folder = r"C:\Users\param\OneDrive\Desktop\Paramspace\trading-files\data"
os.makedirs(data_folder, exist_ok=True)

nino.to_csv(os.path.join(data_folder, "nino_data.csv"), index=False)
coffee.to_csv(os.path.join(data_folder, "coffee_data.csv"), index=False)
merged.to_csv(os.path.join(data_folder, "merged_data.csv"), index=False)

print(f"Data updated in {data_folder}")

from datetime import datetime

merged["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
merged.to_csv(os.path.join(data_folder, "merged_data.csv"), index=False)