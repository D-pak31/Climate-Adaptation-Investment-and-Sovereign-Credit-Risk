###################################################################
# MASTER SCRIPT
# Climate Adaptation & Sovereign Risk
#
# This script:
#  - Cleans raw CDS data
#  - Processes EM-DAT disaster data
#  - Builds the full country-year panel
#  - Runs baseline and heterogeneous regressions
#  - Performs interaction models and an event study
#  - Produces final figures for the paper
###################################################################

import pandas as pd
import numpy as np
import pycountry
import warnings
from linearmodels import PanelOLS
import statsmodels.api as sm
import matplotlib

# Force matplotlib to open figures in a separate window
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns

# Disable future warnings to keep output clean
warnings.simplefilter("ignore", FutureWarning)


# ===============================================================
# 1. PROCESS RAW CDS DATA (DAILY → ANNUAL)
# ===============================================================
def process_cds_data(raw_file="Sovereign_Y_5.xlsx"):
    print("\n[STEP 1] Processing raw CDS data...")

    try:
        df = pd.read_excel(raw_file)
    except FileNotFoundError:
        print(f"ERROR: File {raw_file} not found.")
        return

    # Standardize column names
    df = df.rename(columns={
        "CF_NAME": "country",
        "CF_CURR": "currency",
        "Tenor": "tenor",
        "Date": "date",
        "Par Mid Spread": "cds_spread_bps"
    })

    # Clean string columns and forward-fill missing identifiers
    for col in ["country", "currency", "tenor"]:
        df[col] = df[col].astype("string").str.strip().replace({"": pd.NA}).ffill()

    # Parse dates and CDS values
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cds_spread_bps"] = pd.to_numeric(df["cds_spread_bps"], errors="coerce")
    df = df.dropna(subset=["country", "date"])

    # Keep only USD-denominated 5-year CDS
    df = df[df["currency"].str.upper() == "USD"]
    df = df[df["tenor"].str.contains("5", case=False)]
    df = df[(df["date"].dt.year >= 2009) & (df["date"].dt.year <= 2023)]

    # Annual aggregation
    df["year"] = df["date"].dt.year
    df["cds_spread_bps"] = df["cds_spread_bps"].fillna(
        df.groupby(["country", "year"])["cds_spread_bps"].transform("mean")
    )

    df_median = df.groupby(["country", "year"])["cds_spread_bps"].median().reset_index()
    df_median = df_median.rename(columns={"cds_spread_bps": "Annual_Median_CDS"})

    df_mean = df.groupby(["country", "year"])["cds_spread_bps"].mean().reset_index()
    df_mean = df_mean.rename(columns={"cds_spread_bps": "Annual_Average_CDS"})

    # Convert country names to ISO3 codes
    def get_iso3(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except:
            return None

    # Manual fixes for problematic country names
    manual_map = {
        "BOLIVIA": "BOL",
        "COTE IVOIRE": "CIV",
        "CONGO": "COG",
        "CONGO DEMOCRATIC REPUBLIC": "COD",
        "KOREA REPUBLIC": "KOR",
        "VIET NAM": "VNM",
        "KOSOVO": "XKX",
        "ESWATINI": "SWZ"
    }

    for d in (df_median, df_mean):
        d["Country Code"] = d["country"].apply(get_iso3)
        d["Country Code"] = d.apply(
            lambda r: manual_map.get(r["country"].strip(), r["Country Code"]), axis=1
        )

    # Save processed CDS files
    df_median.to_excel("Annual_Median_CDS.xlsx", index=False)
    df_mean.to_excel("Annual_Average_CDS.xlsx", index=False)

    print(" -> Annual CDS files created.")


# ===============================================================
# 2. PROCESS EM-DAT DISASTER DATA
# ===============================================================
def process_emdat_data(raw_file="Emdat disasterdata.xlsx"):
    print("\n[STEP 2] Processing EM-DAT disaster data...")

    try:
        df = pd.read_excel(raw_file)
    except FileNotFoundError:
        print("ERROR: EM-DAT file not found.")
        return

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

    # Keep only natural disasters
    df = df[df["Disaster_Group"].str.contains("Natural", case=False, na=False)]

    year_col = "Year" if "Year" in df.columns else "Start_Year"
    df = df[(df[year_col] >= 2009) & (df[year_col] <= 2023)]

    # Aggregate disaster indicators at country-year level
    agg_logic = {
        "DisasterCount": ("ISO", "count"),
        "TotalAffected": ("Total_Affected", "sum"),
        "TotalDamage_USD": ("Total_Damage_000_US$", "sum")
    }

    panel = (
        df.groupby(["ISO", "Country", year_col]).agg(**agg_logic)
        .reset_index()
        .rename(columns={"ISO": "Country Code", year_col: "Year"})
    )

    panel.to_csv("EMDAT2009_2023.csv", index=False)
    print(" -> EM-DAT panel saved.")


# ===============================================================
# 3. BUILD MASTER PANEL
# ===============================================================
def create_master_panel():
    print("\n[STEP 3] Building master panel...")

    file_map = {
        "avg_cds": "Annual_Average_CDS.xlsx",
        "med_cds": "Annual_Median_CDS.xlsx",
        "adapt": "AdaptativeInvestment2009-2023.xlsx",
        "vuln": "Vulnerability_index.xlsx",
        "ready": "Readiness_index.xlsx",
        "gdp": "GDPgrowthrate.xlsx",
        "debt": "DebttoGDP.xlsx",
        "infl": "Inflation_%PCI.xlsx",
        "res": "Reserve_USD.xlsx",
    }

    dfs = {name: pd.read_excel(path) for name, path in file_map.items()}
    emdat = pd.read_csv("EMDAT2009_2023.csv")

    # Convert wide country-year datasets into long format
    def reshape(df, newcol):
        df.columns = df.columns.astype(str)
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]
        if "Country Code" not in df.columns and "ISO" in df.columns:
            df = df.rename(columns={"ISO": "Country Code"})
        year_cols = [c for c in df.columns if c.isdigit()]
        df = df[["Country Code"] + year_cols]
        df = df.melt(id_vars="Country Code", var_name="Year", value_name=newcol)
        df["Year"] = df["Year"].astype(int)
        return df

    avg = dfs["avg_cds"].rename(columns={"year": "Year"})
    med = dfs["med_cds"].rename(columns={"year": "Year"})

    adapt = reshape(dfs["adapt"], "AdaptativeInv")
    vuln = reshape(dfs["vuln"], "Vulnerability")
    ready = reshape(dfs["ready"], "Readiness")
    gdp = reshape(dfs["gdp"], "GDPgrowthrate")
    debt = reshape(dfs["debt"], "PublicdebttoGDP")
    infl = reshape(dfs["infl"], "InflationCPI")
    res = reshape(dfs["res"], "ForeignReserve")

    # Merge everything into one panel
    panel = avg.merge(med, on=["Country Code", "Year"], how="outer")
    for d in [adapt, vuln, ready, gdp, debt, infl, res, emdat]:
        panel = panel.merge(d, on=["Country Code", "Year"], how="left")

    panel = panel.dropna(subset=["Annual_Average_CDS", "AdaptativeInv"])

    # Ensure numeric types
    for col in [
        "Annual_Average_CDS", "Annual_Median_CDS", "AdaptativeInv",
        "Vulnerability", "Readiness", "GDPgrowthrate",
        "PublicdebttoGDP", "InflationCPI", "ForeignReserve", "DisasterCount"
    ]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel["DisasterCount"] = panel["DisasterCount"].fillna(0)

    # Log transforms
    panel["ln_AvgCDS"] = np.log(panel["Annual_Average_CDS"])
    panel["ln_MedCDS"] = np.log(panel["Annual_Median_CDS"])
    panel["ln_DebtGDP"] = np.log(panel["PublicdebttoGDP"])

    # Z-score standardization
    def z(x): return (x - x.mean()) / x.std()

    zmap = {
        "AdaptativeInv": "z_AdaptInv",
        "Vulnerability": "z_Vulnerability",
        "Readiness": "z_Readiness",
        "GDPgrowthrate": "z_GDPgrowth",
        "ln_DebtGDP": "z_ln_DebtGDP",
        "InflationCPI": "z_Inflation",
        "ForeignReserve": "z_Reserves"
    }

    for col, newcol in zmap.items():
        panel[newcol] = z(panel[col])

    # Risk classification using average CDS
    mean_cds = panel.groupby("Country Code")["ln_AvgCDS"].mean().reset_index()
    p25, p75 = mean_cds["ln_AvgCDS"].quantile([0.25, 0.75])

    def grp(v):
        if v <= p25: return "Q1_Low"
        if v > p75: return "Q4_High"
        return "Other"

    mean_cds["CDS_Category"] = mean_cds["ln_AvgCDS"].apply(grp)
    panel = panel.merge(mean_cds[["Country Code", "CDS_Category"]], on="Country Code")

    # Lag adaptation by two years
    panel = panel.sort_values(["Country Code", "Year"])
    panel["z_AdaptInv_lag2"] = panel.groupby("Country Code")["z_AdaptInv"].shift(2)
    panel = panel.dropna(subset=["z_AdaptInv_lag2"])

    # Interaction terms
    panel["Int_AdaptLag2_Vulnerability"] = panel["z_AdaptInv_lag2"] * panel["z_Vulnerability"]
    panel["Int_AdaptLag2_Readiness"] = panel["z_AdaptInv_lag2"] * panel["z_Readiness"]

    print(" -> Master panel ready.")
    return panel
# ===============================================================
# SECTION 4 — RUN REGRESSIONS
# ===============================================================
def run_analysis(df):
    print("\n[STEP 4] Running Regressions...")

    df = df.set_index(["Country Code", "Year"])

    base_main = ["z_AdaptInv_lag2", "z_Vulnerability", "z_Readiness"]
    controls = ["z_ln_DebtGDP", "z_GDPgrowth", "z_Inflation", "z_Reserves"]

    interact_vuln = base_main + ["Int_AdaptLag2_Vulnerability"] + controls
    interact_ready = base_main + ["Int_AdaptLag2_Readiness"] + controls

    def run_model(data, y, xvars, title):
        print(f"\n>>> {title}")
        exog = sm.add_constant(data[xvars])
        model = PanelOLS(data[y], exog, entity_effects=True, time_effects=True)
        res = model.fit(cov_type="clustered", cluster_entity=True)
        print(res)
        return res

    # ----------------------------------------------------------
    # 1. BASELINE REGRESSIONS
    # ----------------------------------------------------------
    print("\n--- Baseline: ALL Countries ---")
    run_model(df, "ln_AvgCDS", base_main + controls, "Baseline (All)")

    print("\n--- Baseline: Q1 vs Q4 ---")
    for g in ["Q1_Low", "Q4_High"]:
        run_model(df[df["CDS_Category"] == g],
                  "ln_AvgCDS", base_main + controls,
                  f"Baseline ({g})")

    # ----------------------------------------------------------
    # 2. ROBUSTNESS — MEDIAN CDS
    # ----------------------------------------------------------
    print("\n--- Robustness: Median CDS ---")
    for g in ["Q1_Low", "Q4_High"]:
        run_model(df[df["CDS_Category"] == g],
                  "ln_MedCDS", base_main + controls,
                  f"Robustness Median ({g})")

    # ----------------------------------------------------------
    # 3. INTERACTION MODELS
    # ----------------------------------------------------------
    print("\n====================================================")
    print("   INTERACTION MODELS — ALL COUNTRIES")
    print("====================================================")

    run_model(df,
              "ln_AvgCDS",
              interact_vuln,
              "Interaction: Adaptation × Vulnerability")

    run_model(df,
              "ln_AvgCDS",
              interact_ready,
              "Interaction: Adaptation × Readiness")

    # ----------------------------------------------------------
    # 4. EVENT STUDY — EventID as entity (using DisasterCount ≥ 3)
    # ----------------------------------------------------------
    print("\n======================")
    print("      EVENT STUDY (EventID FE, DisasterCount ≥ 3)")
    print("======================")

    df_evt = df.reset_index().copy()  # 'Country Code' and 'Year'

    # Ensure DisasterCount exists
    if "DisasterCount" not in df_evt.columns:
        df_evt["DisasterCount"] = 0

    # -----------------------------
    # DEFINE SHOCK: DisasterCount ≥ 3
    # -----------------------------
    df_evt["Major_Disaster"] = (df_evt["DisasterCount"] >= 3).astype(int)
    print(f"Shock threshold: DisasterCount ≥ 3")

    # List of event years (one per Country Code × Year)
    events = df_evt[df_evt["Major_Disaster"] == 1][["Country Code", "Year"]].drop_duplicates()

    # Build EventID-based panel: window t-3..t+3
    lags = list(range(-3, 4))
    rows = []

    for _, ev in events.iterrows():
        ccode = ev["Country Code"]
        ey = int(ev["Year"])
        eid = f"{ccode}_{ey}"  # Event ID = country-year of disaster

        for k in lags:
            rows.append({
                "EventID": eid,
                "Country Code": ccode,
                "EventYear": ey,
                "Year": ey + k,
                "rel_time": k
            })

    df_es = pd.DataFrame(rows)

    # Merge with CDS + macro panel
    df_evt_small = df_evt.drop_duplicates(subset=["Country Code", "Year"])
    df_es = df_es.merge(
        df_evt_small,
        on=["Country Code", "Year"],
        how="left"
    )

    # Remove rows without CDS
    df_es = df_es.dropna(subset=["ln_AvgCDS"])

    # -------------------------------------------
    # CREATE EVENT-TIME DUMMIES (DROP t-1 BASELINE)
    # -------------------------------------------
    event_vars = []
    baseline = -1  # omitted dummy

    for k in lags:
        if k == baseline:
            continue
        name = f"t{k:+d}"
        df_es[name] = (df_es["rel_time"] == k).astype(int)
        event_vars.append(name)

    # Index for PanelOLS
    df_es = df_es.set_index(["EventID", "Year"])

    y_es = df_es["ln_AvgCDS"]

    controls_es = [
        "z_Vulnerability",
        "z_Readiness",
        "z_AdaptInv_lag2",
        "z_ln_DebtGDP",
        "z_GDPgrowth",
        "z_Inflation",
        "z_Reserves"
    ]

    X_es = df_es[event_vars + controls_es]
    X_es = X_es.loc[:, X_es.std() > 0]  # remove non-varying variables

    clusters = df_es["Country Code"]

    # ----------------- HIGH-RISK (Q4) -----------------
    print("\n>>> Event Study: High-Risk Countries (Q4, EventID FE)")

    mask_q4 = df_es["CDS_Category"] == "Q4_High"
    y_q4 = y_es[mask_q4]
    X_q4 = X_es[mask_q4]
    clusters_q4 = clusters[mask_q4]

    model_q4 = PanelOLS(
        y_q4,
        X_q4,
        entity_effects=True,  # event FE
        time_effects=True,  # global year FE
        drop_absorbed=True
    )

    res_q4 = model_q4.fit(
        cov_type="clustered",
        clusters=clusters_q4
    )

    print(res_q4)

    # ----------------- LOW-RISK (Q1) -----------------
    print("\n>>> Event Study: Low-Risk Countries (Q1, EventID FE)")

    mask_q1 = df_es["CDS_Category"] == "Q1_Low"
    y_q1 = y_es[mask_q1]
    X_q1 = X_es[mask_q1]
    clusters_q1 = clusters[mask_q1]

    if len(y_q1) > 0 and X_q1.shape[0] > X_q1.shape[1]:
        model_q1 = PanelOLS(
            y_q1,
            X_q1,
            entity_effects=True,
            time_effects=True,
            drop_absorbed=True
        )

        res_q1 = model_q1.fit(
            cov_type="clustered",
            clusters=clusters_q1
        )

        print(res_q1)
    else:
        print(" [!] Not enough Q1 event data to estimate EventID FE model.")


# ===============================================================
# 6. FINAL PLOTS
# ===============================================================

def generate_final_plots(df):

    print("\n===============================")
    print(" STEP 5 — Generating Plots")
    print("===============================\n")

    sns.set_style("whitegrid")
    colors = {'Q4_High': '#d62728', 'Q1_Low': '#2ca02c', 'Other': 'grey'}

    # -------------------------------------------------------
    # PLOT 1: Adaptation Histogram
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(df["AdaptativeInv"], bins=40, kde=True,
                 color="#1f77b4", edgecolor="black", alpha=0.6)
    plt.axvline(df["AdaptativeInv"].mean(), color='red', linestyle='--',
                label=f"Mean = {df['AdaptativeInv'].mean():.2f}")
    plt.title("Distribution of Adaptation Investment (% of GDP)", fontsize=14, fontweight='bold')
    plt.xlabel("Adaptation Investment (% of GDP)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # PLOT 2: Average CDS Trend
    # -------------------------------------------------------
    annual_trend = df.groupby("Year")["Annual_Average_CDS"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(annual_trend.index, annual_trend.values,
             marker="o", linewidth=2, color="navy")
    plt.title("Average Sovereign CDS Spread (2009–2023)", fontsize=14, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel("CDS Spread (bps)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # PLOT 3: Readiness vs Vulnerability
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Vulnerability", y="Readiness",
                    hue="CDS_Category", palette=colors, alpha=0.7, s=80)
    sns.regplot(data=df, x="Vulnerability", y="Readiness",
                scatter=False, color="black", line_kws={"linestyle": "--"})
    plt.title("Development Trap: Vulnerability vs Readiness", fontsize=14, fontweight='bold')
    plt.xlabel("Vulnerability Index")
    plt.ylabel("Readiness Index")
    plt.legend(title="Risk Category")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # PLOT 4: Adaptation Regression (Q1 vs Q4)
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for cat in ["Q4_High", "Q1_Low"]:
        subset = df[df["CDS_Category"] == cat]
        sns.regplot(data=subset, x="z_AdaptInv_lag2", y="ln_AvgCDS",
                    scatter_kws={'alpha': 0.25}, line_kws={'linewidth': 2},
                    color=colors[cat], label=cat)
    plt.title("Adaptation Investment (Lag 2) vs Log CDS Spread", fontsize=14, fontweight='bold')
    plt.xlabel("Adaptation (Z-score, 2-year lag)")
    plt.ylabel("Log CDS Spread")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ======================================================
    #  RESILIENCE PREMIUM PLOT
    # ======================================================

    print("→ Generating Resilience Premium Plot...")

    groups = ["Q4_High", "Q1_Low"]
    labels = ["Vulnerable (Q4)", "Safe (Q1)"]
    betas, errors = [], []

    controls = ["z_AdaptInv_lag2", "z_Vulnerability", "z_Readiness",
                "z_ln_DebtGDP", "z_GDPgrowth", "z_Inflation", "z_Reserves"]

    for cat in groups:
        subset = df[df["CDS_Category"] == cat].set_index(["Country Code", "Year"])
        model = PanelOLS(subset["ln_AvgCDS"],
                         sm.add_constant(subset[controls]),
                         entity_effects=True, time_effects=True)
        res = model.fit(cov_type="clustered", cluster_entity=True)
        betas.append(res.params["z_AdaptInv_lag2"])
        errors.append(res.std_errors["z_AdaptInv_lag2"])

    plt.figure(figsize=(8, 6))
    plt.bar(labels, betas, yerr=errors, capsize=8,
            color=["#d62728", "#2ca02c"], alpha=0.8, edgecolor="black")
    plt.axhline(0, color="black")
    plt.title("Resilience Premium:\nEffect of Adaptation Investment on CDS Spreads", fontsize=14, fontweight='bold')
    plt.ylabel("Coefficient Estimate")
    plt.tight_layout()
    plt.show()

    print("\n✓ All plots successfully generated.\n")

# ===============================================================
# MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    process_cds_data()
    process_emdat_data()
    master = create_master_panel()

    if master is not None:
        # Run the regressions
        run_analysis(master)
        generate_final_plots(master)
    else:
        print("Master panel could not be created.")
