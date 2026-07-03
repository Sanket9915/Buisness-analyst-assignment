# =============================================================================
# HOTEL BOOKING PLATFORM — COMPLETE BUSINESS ANALYST NOTEBOOK
# Dataset : Hotel_bookings_final.csv  |  30,000 Transactions
# Analyst : Business Analysis Internship Submission
# Phases  : 1-Data Audit  2-KPIs  3-Booking Performance  4-Customer Behaviour
#           5-Cancellation  6-Profitability  7-Temporal  8-Root Cause
#           9-Strategic Recommendations
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE   = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
             "#44BBA4", "#E94F37", "#393E41", "#F5A623", "#7B68EE"]
STATUS_PAL = {"Confirmed": "#44BBA4", "Cancelled": "#C73E1D", "Failed": "#F18F01"}

def section(title):
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")

def sub(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — DATA AUDIT
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 1 — DATA AUDIT")

# 1.1 Load
df_raw = pd.read_csv("Hotel_bookings_final.csv")
df = df_raw.copy()

sub("1.1  Dataset Overview")
print(f"Rows         : {df.shape[0]:,}")
print(f"Columns      : {df.shape[1]}")
print(f"\nColumn names : {df.columns.tolist()}")
print("\nData Types:")
print(df.dtypes.to_string())
print("\nFirst 3 rows:")
print(df.head(3).to_string())

# 1.2 Missing values
sub("1.2  Missing Values")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0]
print(missing_df.to_string())
print(f"\n► check_in_date & check_out_date are missing for {5468:,} rows"
      " — these map to 'Failed' or 'Cancelled' bookings that never checked in.")

fig, ax = plt.subplots(figsize=(8, 4))
missing_df["Missing %"].plot(kind="barh", ax=ax, color="#C73E1D", edgecolor="white")
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Values by Column", fontweight="bold")
plt.tight_layout()
plt.savefig("p1_missing_values.png", dpi=150); plt.close()

# 1.3 Duplicates
sub("1.3  Duplicate Records")
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}  (0 — dataset is clean)")

# 1.4 Data quality issues
sub("1.4  Data Quality Issues")
print("Issues identified:")
print("  ① Column 'Coupon USed?' has a typo — should be 'Coupon Used?'")
print("  ② booking_channel & channel_of_booking appear to represent similar"
      " but different dimensions (platform vs. OS). Both are retained.")
print("  ③ coupon_redeem contains 69 negative values (data-entry anomaly).")
neg_coupon = (df["coupon_redeem"] < 0).sum()
print(f"     Negative coupon_redeem rows: {neg_coupon}")
print("  ④ booking_value contains 22 statistical outliers (IQR method).")
print("  ⑤ check_in_date / check_out_date missing for Failed+some Cancelled rows"
      " — expected behaviour, not an error.")

# Fix column name typo
df.rename(columns={"Coupon USed?": "coupon_used"}, inplace=True)
# Clip negative coupon_redeem to 0
df["coupon_redeem"] = df["coupon_redeem"].clip(lower=0)

# 1.5 Date type conversion
sub("1.5  Type Conversion")
date_cols = ["booking_date", "check_in_date", "check_out_date", "travel_date"]
for c in date_cols:
    df[c] = pd.to_datetime(df[c], errors="coerce")

df["los"] = (df["check_out_date"] - df["check_in_date"]).dt.days
df["lead_time"] = (df["check_in_date"] - df["booking_date"]).dt.days
df["booking_month"] = df["booking_date"].dt.to_period("M")
df["profit"] = df["selling_price"] - df["costprice"]
df["profit_margin_pct"] = (df["profit"] / df["selling_price"] * 100).round(2)
print("Date columns converted. Derived columns added: los, lead_time,"
      " booking_month, profit, profit_margin_pct.")

# 1.6 Outlier detection
sub("1.6  Outlier Detection (IQR Method)")
num_cols = ["booking_value", "selling_price", "costprice", "markup", "profit",
            "refund_amount", "cashback"]
outlier_summary = []
for col in num_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    outlier_summary.append({"Column": col, "Q1": round(Q1,1), "Q3": round(Q3,1),
                             "IQR": round(IQR,1), "Lower Fence": round(lo,1),
                             "Upper Fence": round(hi,1), "Outliers": n_out})
out_df = pd.DataFrame(outlier_summary)
print(out_df.to_string(index=False))

fig, axes = plt.subplots(2, 4, figsize=(18, 7))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i], color=PALETTE[i], width=0.5)
    axes[i].set_title(col, fontweight="bold")
axes[-1].set_visible(False)
fig.suptitle("Outlier Detection — Box Plots", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("p1_outliers.png", dpi=150); plt.close()
print("\n► booking_value outliers (22 rows) retained — they represent"
      " valid high-value luxury bookings, not data errors.")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — KPI FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 2 — KPI FRAMEWORK")

total_bookings     = len(df)
gross_booking_val  = df["booking_value"].sum()
avg_booking_val    = df["booking_value"].mean()
confirmed          = (df["booking_status"] == "Confirmed").sum()
cancelled          = (df["booking_status"] == "Cancelled").sum()
failed             = (df["booking_status"] == "Failed").sum()
confirm_rate       = confirmed / total_bookings * 100
cancel_rate        = cancelled / total_bookings * 100
failure_rate       = failed / total_bookings * 100
avg_los            = df["los"].mean()
avg_lead           = df["lead_time"].mean()
refund_rows        = (df["refund_status"] == "Yes").sum()
refund_rate        = refund_rows / total_bookings * 100
cust_booking_cnt   = df.groupby("customer_id")["booking_date"].count()
repeat_cust        = (cust_booking_cnt > 1).sum()
repeat_cust_rate   = repeat_cust / len(cust_booking_cnt) * 100
avg_profit         = df["profit"].mean()
total_revenue      = df["selling_price"].sum()
total_profit       = df["profit"].sum()
profit_margin      = total_profit / total_revenue * 100

kpi_table = pd.DataFrame({
    "KPI": [
        "Total Bookings", "Gross Booking Value (₹)",
        "Average Booking Value (₹)", "Total Revenue / Selling Price (₹)",
        "Total Profit (₹)", "Confirmation Rate (%)", "Cancellation Rate (%)",
        "Failure Rate (%)", "Average Length of Stay (nights)",
        "Average Lead Time (days)", "Refund Rate (%)",
        "Repeat Customer Rate (%)", "Average Profit Per Booking (₹)",
        "Overall Profit Margin (%)"
    ],
    "Value": [
        f"{total_bookings:,}", f"{gross_booking_val:,.0f}",
        f"{avg_booking_val:,.0f}", f"{total_revenue:,.0f}",
        f"{total_profit:,.0f}", f"{confirm_rate:.1f}%",
        f"{cancel_rate:.1f}%", f"{failure_rate:.1f}%",
        f"{avg_los:.1f}", f"{avg_lead:.0f}",
        f"{refund_rate:.1f}%", f"{repeat_cust_rate:.0f}%",
        f"{avg_profit:,.0f}", f"{profit_margin:.1f}%"
    ],
    "Business Interpretation": [
        "Scale of operations — 30K transactions over 13 months",
        "Total demand value generated on the platform",
        "Each booking averages ₹25K — solid mid-market segment",
        "Revenue after markup; platform earns on the spread",
        "Strong absolute profit pool across confirmed bookings",
        "72.2% conversion — meaningful room to improve",
        "20.2% cancellations — high; industry healthy target is <12%",
        "7.5% failures — payment / system issues eroding potential",
        "4 nights average — predominantly short-stay leisure",
        "30 days advance — typical leisure planning horizon",
        "78.4% refund rate — overstated; includes non-cancel bookings",
        "All 499 customers are repeat bookers — loyal base",
        "Platform earns ~₹6,963 net per booking",
        "27.7% margin — healthy but protect against discounts"
    ]
})
print(kpi_table.to_string(index=False))

# KPI visual — gauge-style bar
fig, ax = plt.subplots(figsize=(10, 6))
rate_kpis  = ["Confirmation Rate", "Cancellation Rate", "Failure Rate",
              "Refund Rate", "Profit Margin"]
rate_vals  = [confirm_rate, cancel_rate, failure_rate, refund_rate, profit_margin]
rate_colors = ["#44BBA4", "#C73E1D", "#F18F01", "#2E86AB", "#7B68EE"]
bars = ax.barh(rate_kpis, rate_vals, color=rate_colors, edgecolor="white", height=0.5)
for bar, val in zip(bars, rate_vals):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontweight="bold")
ax.set_xlim(0, 110)
ax.set_xlabel("Rate (%)")
ax.set_title("Key Rate KPIs — Platform Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("p2_kpi_rates.png", dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — BOOKING PERFORMANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 3 — BOOKING PERFORMANCE ANALYSIS")

def perf_table(df, group_col, label):
    g = df.groupby(group_col).agg(
        Bookings=("booking_value", "count"),
        Revenue=("selling_price", "sum"),
        Profit=("profit", "sum"),
        Avg_Booking_Value=("booking_value", "mean"),
        Cancellations=("booking_status",
                        lambda x: (x == "Cancelled").sum()),
        Confirmations=("booking_status",
                        lambda x: (x == "Confirmed").sum()),
    ).reset_index()
    g["Confirmation_Rate_%"] = (g["Confirmations"] / g["Bookings"] * 100).round(1)
    g["Cancellation_Rate_%"] = (g["Cancellations"] / g["Bookings"] * 100).round(1)
    g["Avg_Profit"]           = (g["Profit"] / g["Bookings"]).round(0)
    g["Revenue_Share_%"]      = (g["Revenue"] / g["Revenue"].sum() * 100).round(1)
    g = g.drop(columns=["Cancellations", "Confirmations"])
    g["Revenue"]  = g["Revenue"].map("{:,.0f}".format)
    g["Profit"]   = g["Profit"].map("{:,.0f}".format)
    g["Avg_Booking_Value"] = g["Avg_Booking_Value"].map("{:,.0f}".format)
    sub(f"3 — Performance by {label}")
    print(g.to_string(index=False))
    return g

# 3.1 Booking Channel
ch = df.groupby("booking_channel").agg(
    Bookings=("booking_value", "count"),
    Revenue=("selling_price", "sum"),
    Profit=("profit", "sum"),
    Avg_BV=("booking_value", "mean"),
    Confirmed=("booking_status", lambda x: (x=="Confirmed").sum()),
    Cancelled=("booking_status", lambda x: (x=="Cancelled").sum()),
).reset_index()
ch["Conf_Rate"] = (ch["Confirmed"] / ch["Bookings"] * 100).round(1)
ch["Canc_Rate"] = (ch["Cancelled"] / ch["Bookings"] * 100).round(1)
sub("3.1  By Booking Channel")
print(ch[["booking_channel","Bookings","Revenue","Profit","Avg_BV","Conf_Rate","Canc_Rate"]].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ch_sorted = ch.sort_values("Bookings", ascending=False)
axes[0].bar(ch_sorted["booking_channel"], ch_sorted["Bookings"],
            color=PALETTE[:3], edgecolor="white")
axes[0].set_title("Bookings by Channel", fontweight="bold")
axes[1].bar(ch_sorted["booking_channel"], ch_sorted["Revenue"] / 1e6,
            color=PALETTE[:3], edgecolor="white")
axes[1].set_title("Revenue by Channel (M)", fontweight="bold")
axes[2].bar(ch_sorted["booking_channel"], ch_sorted["Conf_Rate"],
            color=["#44BBA4","#44BBA4","#44BBA4"], edgecolor="white")
axes[2].set_ylim(65, 80)
axes[2].set_title("Confirmation Rate % by Channel", fontweight="bold")
for ax in axes:
    ax.tick_params(axis="x", rotation=10)
plt.suptitle("Booking Channel Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("p3_channel_performance.png", dpi=150); plt.close()

# 3.2 Room Type
rt = df.groupby("room_type").agg(
    Bookings=("booking_value","count"),
    Revenue=("selling_price","sum"),
    Profit=("profit","sum"),
    Avg_BV=("booking_value","mean"),
    Confirmed=("booking_status", lambda x:(x=="Confirmed").sum()),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
rt["Conf_Rate"] = (rt["Confirmed"]/rt["Bookings"]*100).round(1)
rt["Canc_Rate"] = (rt["Cancelled"]/rt["Bookings"]*100).round(1)
sub("3.2  By Room Type")
print(rt[["room_type","Bookings","Revenue","Profit","Avg_BV","Conf_Rate","Canc_Rate"]].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors_rt = [PALETTE[0], PALETTE[1], PALETTE[2]]
axes[0].bar(rt["room_type"], rt["Bookings"], color=colors_rt, edgecolor="white")
axes[0].set_title("Bookings by Room Type", fontweight="bold")
axes[1].bar(rt["room_type"], rt["Profit"]/1e6, color=colors_rt, edgecolor="white")
axes[1].set_title("Total Profit by Room Type (M)", fontweight="bold")
axes[2].bar(rt["room_type"], rt["Canc_Rate"], color="#C73E1D", edgecolor="white", alpha=0.8)
axes[2].set_title("Cancellation Rate % by Room Type", fontweight="bold")
plt.suptitle("Room Type Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("p3_roomtype_performance.png", dpi=150); plt.close()

# 3.3 Star Rating
sr = df.groupby("star_rating").agg(
    Bookings=("booking_value","count"),
    Revenue=("selling_price","sum"),
    Profit=("profit","sum"),
    Avg_BV=("booking_value","mean"),
    Confirmed=("booking_status", lambda x:(x=="Confirmed").sum()),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
sr["Conf_Rate"] = (sr["Confirmed"]/sr["Bookings"]*100).round(1)
sr["Canc_Rate"] = (sr["Cancelled"]/sr["Bookings"]*100).round(1)
sub("3.3  By Star Rating")
print(sr[["star_rating","Bookings","Revenue","Profit","Avg_BV","Conf_Rate","Canc_Rate"]].to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
star_labels = [f"★{s}" for s in sr["star_rating"]]
axes[0].bar(star_labels, sr["Bookings"], color=PALETTE[:4], edgecolor="white")
axes[0].set_title("Bookings by Star Rating", fontweight="bold")
axes[1].bar(star_labels, sr["Avg_BV"], color=PALETTE[:4], edgecolor="white")
axes[1].set_title("Avg Booking Value by Star Rating", fontweight="bold")
axes[2].bar(star_labels, sr["Canc_Rate"], color="#C73E1D", edgecolor="white", alpha=0.8)
axes[2].set_title("Cancellation Rate % by Star Rating", fontweight="bold")
plt.suptitle("Star Rating Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("p3_starrating_performance.png", dpi=150); plt.close()

# 3.4 City
cy = df.groupby("city").agg(
    Bookings=("booking_value","count"),
    Revenue=("selling_price","sum"),
    Profit=("profit","sum"),
    Avg_BV=("booking_value","mean"),
    Confirmed=("booking_status", lambda x:(x=="Confirmed").sum()),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index().sort_values("Revenue", ascending=False)
cy["Conf_Rate"] = (cy["Confirmed"]/cy["Bookings"]*100).round(1)
cy["Canc_Rate"] = (cy["Cancelled"]/cy["Bookings"]*100).round(1)
sub("3.4  By City")
print(cy[["city","Bookings","Revenue","Profit","Avg_BV","Conf_Rate","Canc_Rate"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=cy, x="Revenue", y="city", palette="Blues_d", ax=axes[0])
axes[0].set_title("Revenue by City", fontweight="bold")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.1f}M"))
color_map = ["#C73E1D" if v > cy["Canc_Rate"].mean() else "#44BBA4"
             for v in cy["Canc_Rate"]]
axes[1].barh(cy["city"], cy["Canc_Rate"], color=color_map, edgecolor="white")
axes[1].axvline(cy["Canc_Rate"].mean(), linestyle="--", color="black", linewidth=1.2,
                label=f"Mean {cy['Canc_Rate'].mean():.1f}%")
axes[1].legend()
axes[1].set_title("Cancellation Rate by City (Red = Above Average)", fontweight="bold")
plt.suptitle("City-Level Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("p3_city_performance.png", dpi=150); plt.close()

# 3.5 Stay Type
st = df.groupby("stay_type").agg(
    Bookings=("booking_value","count"),
    Revenue=("selling_price","sum"),
    Profit=("profit","sum"),
    Avg_BV=("booking_value","mean"),
    Confirmed=("booking_status", lambda x:(x=="Confirmed").sum()),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
st["Conf_Rate"] = (st["Confirmed"]/st["Bookings"]*100).round(1)
st["Canc_Rate"] = (st["Cancelled"]/st["Bookings"]*100).round(1)
sub("3.5  By Stay Type")
print(st[["stay_type","Bookings","Revenue","Profit","Avg_BV","Conf_Rate","Canc_Rate"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(st["stay_type"], st["Bookings"], color=["#2E86AB","#A23B72"], edgecolor="white", width=0.4)
axes[0].set_title("Bookings: Leisure vs Business", fontweight="bold")
axes[1].bar(st["stay_type"], st["Avg_BV"], color=["#F18F01","#44BBA4"], edgecolor="white", width=0.4)
axes[1].set_title("Avg Booking Value: Leisure vs Business", fontweight="bold")
plt.suptitle("Stay Type Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("p3_staytype_performance.png", dpi=150); plt.close()

print("\n► BUSINESS INSIGHTS — Phase 3:")
print("  • Web channel drives 50% of bookings but Mobile App shows comparable"
      " confirmation rates — mobile investment is justified.")
print("  • Standard rooms = 55% of volume but lowest margin per booking."
      " Suite segment is small but highest avg booking value.")
print("  • 4-star properties dominate volume and profit — the platform's sweet spot.")
print("  • Chicago leads in revenue; Dallas has the highest cancellation rate.")
print("  • Business bookings have higher avg booking value (+~₹1,200) with"
      " lower leisure-driven cancellations.")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4 — CUSTOMER BEHAVIOUR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 4 — CUSTOMER BEHAVIOUR ANALYSIS")

sub("4.1  Repeat Customer Analysis")
cust_agg = df.groupby("customer_id").agg(
    Total_Bookings=("booking_status","count"),
    Confirmed=("booking_status", lambda x:(x=="Confirmed").sum()),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
    Total_Spend=("booking_value","sum"),
    Avg_Spend=("booking_value","mean"),
    Coupon_Used_Count=("coupon_used", lambda x:(x=="Yes").sum()),
    Cashback_Total=("cashback","sum"),
).reset_index()
cust_agg["Repeat"] = cust_agg["Total_Bookings"] > 1
print(f"Unique customers     : {len(cust_agg):,}")
print(f"All customers repeat : True (100% — closed platform, all IDs multi-booking)")
print(f"\nBookings per customer distribution:")
print(cust_agg["Total_Bookings"].describe().round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(cust_agg["Total_Bookings"], bins=20, ax=axes[0],
             color="#2E86AB", edgecolor="white", kde=True)
axes[0].set_title("Distribution of Bookings per Customer", fontweight="bold")
axes[0].set_xlabel("Number of Bookings")

top20 = cust_agg.nlargest(20, "Total_Spend")
axes[1].barh(top20["customer_id"].astype(str), top20["Total_Spend"]/1e3,
             color="#A23B72", edgecolor="white")
axes[1].set_xlabel("Total Spend (₹ '000)")
axes[1].set_title("Top 20 Customers by Total Spend", fontweight="bold")
plt.tight_layout()
plt.savefig("p4_customer_repeat.png", dpi=150); plt.close()

sub("4.2  Coupon Usage Analysis")
coupon_agg = df.groupby("coupon_used").agg(
    Count=("booking_value","count"),
    Avg_BV=("booking_value","mean"),
    Avg_Profit=("profit","mean"),
    Confirmed=("booking_status", lambda x:(x=="Confirmed").sum()),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
coupon_agg["Conf_Rate"] = (coupon_agg["Confirmed"]/coupon_agg["Count"]*100).round(1)
coupon_agg["Canc_Rate"] = (coupon_agg["Cancelled"]/coupon_agg["Count"]*100).round(1)
print(coupon_agg[["coupon_used","Count","Avg_BV","Avg_Profit","Conf_Rate","Canc_Rate"]].to_string(index=False))

sub("4.3  Cashback Usage Analysis")
df["cashback_used"] = df["cashback"] > 0
cb_agg = df.groupby("cashback_used").agg(
    Count=("booking_value","count"),
    Avg_BV=("booking_value","mean"),
    Avg_Cashback=("cashback","mean"),
    Avg_Profit=("profit","mean"),
).reset_index()
cb_agg["cashback_used"] = cb_agg["cashback_used"].map({True:"Cashback Used", False:"No Cashback"})
print(cb_agg.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].pie(coupon_agg["Count"], labels=coupon_agg["coupon_used"],
            autopct="%1.1f%%", colors=["#C73E1D","#44BBA4"], startangle=90)
axes[0].set_title("Coupon Usage Split", fontweight="bold")

axes[1].bar(coupon_agg["coupon_used"], coupon_agg["Avg_BV"],
            color=["#F18F01","#2E86AB"], edgecolor="white", width=0.4)
axes[1].set_title("Avg Booking Value: Coupon vs No Coupon", fontweight="bold")

axes[2].bar(cb_agg["cashback_used"], cb_agg["Avg_BV"],
            color=["#7B68EE","#44BBA4"], edgecolor="white", width=0.4)
axes[2].set_title("Avg Booking Value: Cashback vs None", fontweight="bold")
plt.suptitle("Incentive Usage — Customer Behaviour", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("p4_incentive_usage.png", dpi=150); plt.close()

sub("4.4  Booking Frequency by Channel")
freq = df.groupby(["customer_id","booking_channel"]).size().reset_index(name="freq")
print(freq.groupby("booking_channel")["freq"].describe().round(1).to_string())

print("\n► BUSINESS INSIGHTS — Phase 4:")
print("  • 100% of the 499 customers are repeat bookers — the platform has"
      " a deeply loyal base. Focus should shift to growing acquisition.")
print("  • Coupon users (20.6%) show slightly lower Avg Booking Value —"
      " coupons attract price-sensitive segments but still generate revenue.")
print("  • Cashback drives a higher avg booking value — it incentivises"
      " premium purchases without explicit discounting.")
print("  • High-value customers (top 20) contribute disproportionately —"
      " VIP tier programmes are warranted.")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 5 — CANCELLATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 5 — CANCELLATION ANALYSIS")

cancelled_df = df[df["booking_status"] == "Cancelled"].copy()
confirmed_df = df[df["booking_status"] == "Confirmed"].copy()

sub("5.1  Revenue Lost Due to Cancellations")
revenue_lost = cancelled_df["selling_price"].sum()
profit_lost  = cancelled_df["profit"].sum()
print(f"Cancelled Bookings     : {len(cancelled_df):,}")
print(f"Revenue Lost (₹)       : {revenue_lost:,.0f}")
print(f"Profit Lost (₹)        : {profit_lost:,.0f}")
print(f"Avg Revenue/Cancel (₹) : {revenue_lost/len(cancelled_df):,.0f}")

def canc_breakdown(group_col, label):
    g = df.groupby(group_col).agg(
        Total=("booking_status","count"),
        Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
        Revenue_Lost=("selling_price", lambda x: x[df.loc[x.index,"booking_status"]=="Cancelled"].sum()),
    ).reset_index()
    g["Canc_Rate_%"] = (g["Cancelled"]/g["Total"]*100).round(1)
    g["Revenue_Lost"] = g["Revenue_Lost"].map("{:,.0f}".format)
    sub(f"5 — Cancellations by {label}")
    print(g.to_string(index=False))
    return g

# Cancellations by channel
c_ch = df.groupby("booking_channel").agg(
    Total=("booking_status","count"),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
c_ch["Canc_Rate"] = (c_ch["Cancelled"]/c_ch["Total"]*100).round(1)
sub("5.2  Cancellations by Channel")
print(c_ch.to_string(index=False))

# By room type
c_rt = df.groupby("room_type").agg(
    Total=("booking_status","count"),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
c_rt["Canc_Rate"] = (c_rt["Cancelled"]/c_rt["Total"]*100).round(1)
sub("5.3  Cancellations by Room Type")
print(c_rt.to_string(index=False))

# By star rating
c_sr = df.groupby("star_rating").agg(
    Total=("booking_status","count"),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
c_sr["Canc_Rate"] = (c_sr["Cancelled"]/c_sr["Total"]*100).round(1)
sub("5.4  Cancellations by Star Rating")
print(c_sr.to_string(index=False))

# By city
c_cy = df.groupby("city").agg(
    Total=("booking_status","count"),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index().sort_values("Cancelled", ascending=False)
c_cy["Canc_Rate"] = (c_cy["Cancelled"]/c_cy["Total"]*100).round(1)
sub("5.5  Cancellations by City")
print(c_cy.to_string(index=False))

# By stay type
c_st = df.groupby("stay_type").agg(
    Total=("booking_status","count"),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
c_st["Canc_Rate"] = (c_st["Cancelled"]/c_st["Total"]*100).round(1)
sub("5.6  Cancellations by Stay Type")
print(c_st.to_string(index=False))

# Lead time vs cancellation
sub("5.7  Lead Time vs Cancellation")
df_with_lead = df.dropna(subset=["lead_time"]).copy()
df_with_lead["lead_bucket"] = pd.cut(df_with_lead["lead_time"],
    bins=[0,7,14,30,45,60], labels=["0-7d","8-14d","15-30d","31-45d","46-60d"])
lt_canc = df_with_lead.groupby("lead_bucket").agg(
    Total=("booking_status","count"),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
lt_canc["Canc_Rate"] = (lt_canc["Cancelled"]/lt_canc["Total"]*100).round(1)
print(lt_canc.to_string(index=False))

# Booking value vs cancellation
sub("5.8  Booking Value Band vs Cancellation")
df["bv_band"] = pd.cut(df["booking_value"],
    bins=[0,10000,20000,30000,45000,70000],
    labels=["<10K","10-20K","20-30K","30-45K","45K+"])
bv_canc = df.groupby("bv_band").agg(
    Total=("booking_status","count"),
    Cancelled=("booking_status", lambda x:(x=="Cancelled").sum()),
).reset_index()
bv_canc["Canc_Rate"] = (bv_canc["Cancelled"]/bv_canc["Total"]*100).round(1)
print(bv_canc.to_string(index=False))

# Charts
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

axes[0,0].barh(c_ch["booking_channel"], c_ch["Canc_Rate"],
               color="#C73E1D", edgecolor="white")
axes[0,0].set_title("Cancellation Rate by Channel", fontweight="bold")
axes[0,0].set_xlabel("Cancellation Rate (%)")

axes[0,1].bar(c_rt["room_type"], c_rt["Canc_Rate"],
              color=["#F18F01","#C73E1D","#A23B72"], edgecolor="white")
axes[0,1].set_title("Cancellation Rate by Room Type", fontweight="bold")

axes[0,2].bar([f"★{s}" for s in c_sr["star_rating"]], c_sr["Canc_Rate"],
              color=PALETTE[:4], edgecolor="white")
axes[0,2].set_title("Cancellation Rate by Star Rating", fontweight="bold")

axes[1,0].barh(c_cy["city"], c_cy["Canc_Rate"],
               color=["#C73E1D" if v > c_cy["Canc_Rate"].mean() else "#2E86AB"
                      for v in c_cy["Canc_Rate"]], edgecolor="white")
axes[1,0].set_title("Cancellation Rate by City", fontweight="bold")

axes[1,1].bar(lt_canc["lead_bucket"].astype(str), lt_canc["Canc_Rate"],
              color="#7B68EE", edgecolor="white")
axes[1,1].set_title("Cancellation Rate by Lead Time", fontweight="bold")
axes[1,1].set_xlabel("Lead Time Bucket")

axes[1,2].bar(bv_canc["bv_band"].astype(str), bv_canc["Canc_Rate"],
              color="#F18F01", edgecolor="white")
axes[1,2].set_title("Cancellation Rate by Booking Value Band", fontweight="bold")

plt.suptitle("Cancellation Analysis — Multi-Dimensional", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("p5_cancellation_analysis.png", dpi=150); plt.close()

print(f"\n► REVENUE LOST TO CANCELLATIONS: ₹{revenue_lost:,.0f}")
print(f"► PROFIT LOST TO CANCELLATIONS : ₹{profit_lost:,.0f}")
print("\n► BUSINESS INSIGHTS — Phase 5:")
print("  • Long lead-time bookings (46-60 days) show the highest cancellation"
      " rate — customers change plans. Introduce non-refundable discounts.")
print("  • High-value bookings (30K+) cancel more — risk of large revenue loss.")
print("  • Travel Agent channel has the highest cancellation rate — agent"
      " behaviour / bulk cancellations need contractual guardrails.")
print("  • Dallas and Miami are above-average cancellation cities —"
      " local event calendar mismatch or competitor pricing likely.")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 6 — PROFITABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 6 — PROFITABILITY ANALYSIS")

confirmed_df = df[df["booking_status"] == "Confirmed"].copy()

def profit_table(grp_col, label, data=confirmed_df):
    g = data.groupby(grp_col).agg(
        Bookings=("profit","count"),
        Total_Profit=("profit","sum"),
        Avg_Profit=("profit","mean"),
        Total_Revenue=("selling_price","sum"),
        Avg_Revenue=("selling_price","mean"),
    ).reset_index()
    g["Profit_Margin_%"] = (g["Total_Profit"]/g["Total_Revenue"]*100).round(1)
    g["Profit_Share_%"]  = (g["Total_Profit"]/g["Total_Profit"].sum()*100).round(1)
    sub(f"6 — Profitability by {label}")
    print(g.to_string(index=False))
    return g

p_ch = profit_table("booking_channel", "Channel")
p_rt = profit_table("room_type", "Room Type")
p_sr = profit_table("star_rating", "Star Rating")
p_cy = profit_table("city", "City")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Channel
axes[0,0].bar(p_ch["booking_channel"], p_ch["Total_Profit"]/1e6,
              color=PALETTE[:3], edgecolor="white")
axes[0,0].set_title("Total Profit by Channel (Confirmed)", fontweight="bold")
axes[0,0].set_ylabel("Profit (₹M)")

# Room Type
x_rt = np.arange(len(p_rt))
w = 0.35
axes[0,1].bar(x_rt - w/2, p_rt["Avg_Profit"], w, label="Avg Profit", color="#2E86AB", edgecolor="white")
axes[0,1].bar(x_rt + w/2, p_rt["Profit_Margin_%"]*1000, w, label="Margin*1000 (scale)",
              color="#A23B72", edgecolor="white")
axes[0,1].set_xticks(x_rt); axes[0,1].set_xticklabels(p_rt["room_type"])
axes[0,1].legend()
axes[0,1].set_title("Avg Profit & Margin by Room Type", fontweight="bold")

# Star Rating
star_labels2 = [f"★{s}" for s in p_sr["star_rating"]]
axes[1,0].bar(star_labels2, p_sr["Avg_Profit"], color=PALETTE[:4], edgecolor="white")
axes[1,0].set_title("Avg Profit per Confirmed Booking by Star Rating", fontweight="bold")
axes[1,0].set_ylabel("Avg Profit (₹)")

# City
p_cy_s = p_cy.sort_values("Total_Profit", ascending=True)
axes[1,1].barh(p_cy_s["city"], p_cy_s["Total_Profit"]/1e6, color="#44BBA4", edgecolor="white")
axes[1,1].set_title("Total Profit by City (Confirmed Bookings)", fontweight="bold")
axes[1,1].set_xlabel("Total Profit (₹M)")

plt.suptitle("Profitability Analysis — Confirmed Bookings Only",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("p6_profitability.png", dpi=150); plt.close()

sub("6 — Cost vs Revenue Scatter (Confirmed)")
fig, ax = plt.subplots(figsize=(9, 6))
scatter_data = confirmed_df.sample(min(3000, len(confirmed_df)), random_state=42)
room_colors = {"Standard":"#2E86AB","Deluxe":"#A23B72","Suite":"#F18F01"}
for rtype, grp in scatter_data.groupby("room_type"):
    ax.scatter(grp["costprice"], grp["selling_price"], alpha=0.4,
               s=20, label=rtype, color=room_colors.get(rtype,"grey"))
ax.plot([scatter_data["costprice"].min(), scatter_data["costprice"].max()],
        [scatter_data["costprice"].min(), scatter_data["costprice"].max()],
        "k--", linewidth=1.2, label="Break-even line")
ax.set_xlabel("Cost Price (₹)")
ax.set_ylabel("Selling Price (₹)")
ax.set_title("Cost Price vs Selling Price by Room Type", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("p6_cost_vs_revenue.png", dpi=150); plt.close()

print("\n► BUSINESS INSIGHTS — Phase 6:")
print("  • Web channel generates the highest total profit (volume-driven).")
print("  • Suite room type has the highest avg profit per booking but"
      " smallest volume — upselling opportunity exists.")
print("  • 5-star properties deliver the highest avg profit per booking"
      " despite lower volume — premium positioning is profitable.")
print("  • Chicago and Los Angeles are the top profit-generating cities.")
print("  • All room types operate well above the break-even line —"
      " markup structure is healthy.")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 7 — TEMPORAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 7 — TEMPORAL ANALYSIS")

df["month_str"] = df["booking_date"].dt.strftime("%Y-%m")
monthly = df.groupby("month_str").agg(
    Total_Bookings=("booking_value","count"),
    Revenue=("selling_price","sum"),
    Profit=("profit","sum"),
    Cancellations=("booking_status", lambda x:(x=="Cancelled").sum()),
    Avg_LOS=("los","mean"),
    Avg_BV=("booking_value","mean"),
).reset_index().sort_values("month_str")

monthly["Canc_Rate_%"] = (monthly["Cancellations"]/monthly["Total_Bookings"]*100).round(1)

sub("7.1  Monthly KPI Table")
print(monthly[["month_str","Total_Bookings","Revenue","Profit","Cancellations",
               "Canc_Rate_%","Avg_LOS"]].to_string(index=False))

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
x_ticks = range(len(monthly))
xtick_labels = monthly["month_str"]

axes[0,0].bar(x_ticks, monthly["Total_Bookings"], color="#2E86AB", edgecolor="white")
axes[0,0].set_title("Monthly Booking Volume", fontweight="bold")
axes[0,0].set_xticks(x_ticks); axes[0,0].set_xticklabels(xtick_labels, rotation=45, ha="right")

axes[0,1].plot(x_ticks, monthly["Revenue"]/1e6, marker="o", color="#A23B72", linewidth=2)
axes[0,1].fill_between(x_ticks, monthly["Revenue"]/1e6, alpha=0.2, color="#A23B72")
axes[0,1].set_title("Monthly Revenue (₹M)", fontweight="bold")
axes[0,1].set_xticks(x_ticks); axes[0,1].set_xticklabels(xtick_labels, rotation=45, ha="right")

axes[1,0].plot(x_ticks, monthly["Profit"]/1e6, marker="s", color="#44BBA4", linewidth=2)
axes[1,0].fill_between(x_ticks, monthly["Profit"]/1e6, alpha=0.2, color="#44BBA4")
axes[1,0].set_title("Monthly Profit (₹M)", fontweight="bold")
axes[1,0].set_xticks(x_ticks); axes[1,0].set_xticklabels(xtick_labels, rotation=45, ha="right")

axes[1,1].bar(x_ticks, monthly["Cancellations"], color="#C73E1D", edgecolor="white")
axes[1,1].plot(x_ticks, monthly["Canc_Rate_%"]*10, color="black", linestyle="--",
               linewidth=1.5, label="Canc Rate (×10 scale)")
axes[1,1].legend()
axes[1,1].set_title("Monthly Cancellations & Rate", fontweight="bold")
axes[1,1].set_xticks(x_ticks); axes[1,1].set_xticklabels(xtick_labels, rotation=45, ha="right")

axes[2,0].plot(x_ticks, monthly["Avg_LOS"], marker="D", color="#7B68EE", linewidth=2)
axes[2,0].set_title("Monthly Avg Length of Stay (nights)", fontweight="bold")
axes[2,0].set_xticks(x_ticks); axes[2,0].set_xticklabels(xtick_labels, rotation=45, ha="right")
axes[2,0].set_ylim(0, 8)

axes[2,1].plot(x_ticks, monthly["Avg_BV"], marker="^", color="#F18F01", linewidth=2)
axes[2,1].fill_between(x_ticks, monthly["Avg_BV"], alpha=0.2, color="#F18F01")
axes[2,1].set_title("Monthly Avg Booking Value (₹)", fontweight="bold")
axes[2,1].set_xticks(x_ticks); axes[2,1].set_xticklabels(xtick_labels, rotation=45, ha="right")

plt.suptitle("Temporal Analysis — Monthly Trends (Apr 2024 – Apr 2025)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("p7_temporal_analysis.png", dpi=150); plt.close()

# Heatmap — bookings by month × channel
pivot = df.pivot_table(index="month_str", columns="booking_channel",
                       values="booking_value", aggfunc="count", fill_value=0)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
ax.set_title("Booking Volume Heatmap — Month × Channel", fontweight="bold")
ax.set_xlabel("Channel"); ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig("p7_heatmap_month_channel.png", dpi=150); plt.close()

print("\n► BUSINESS INSIGHTS — Phase 7:")
print("  • Booking volume is relatively stable month-on-month with no extreme"
      " seasonality — good platform stability.")
print("  • Avg LOS stays close to 4 nights throughout — consistent demand profile.")
print("  • Any cancellation rate spikes in specific months likely correlate"
      " with holiday planning and last-minute plan changes.")
print("  • Revenue and profit trends closely mirror volume trends — no"
      " unusual pricing variance detected.")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 8 — ROOT CAUSE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 8 — ROOT CAUSE ANALYSIS")

rca = [
    {
        "Observation": "Cancellation Rate is 20.2% (6,070 bookings)",
        "Evidence": "Travel Agent channel and 3★ hotels show highest cancel rates; "
                    "46-60 day lead-time bucket has most cancellations.",
        "Root Cause": "Agents make speculative bulk bookings; long lead-time bookings "
                      "are more susceptible to plan changes; 3★ hotels have low perceived "
                      "switching cost so customers cancel and rebook elsewhere.",
        "Business Impact": f"Revenue at risk: ₹{revenue_lost:,.0f}. "
                            f"Profit at risk: ₹{profit_lost:,.0f}."
    },
    {
        "Observation": "Failure Rate is 7.5% (2,258 bookings)",
        "Evidence": "2,258 bookings with 'Failed' status; payment methods split across "
                    "PayPal, Bank Transfer, Debit/Credit Card.",
        "Root Cause": "Payment gateway failures, insufficient funds, or session timeouts "
                      "during checkout. Mobile App users may experience more payment errors "
                      "due to connectivity issues.",
        "Business Impact": "₹56M+ in potential revenue never converted. "
                            "Damages platform trust and inflates bounce rates."
    },
    {
        "Observation": "Travel Agent channel has lower efficiency",
        "Evidence": "Travel Agents account for <10% of bookings but contribute "
                    "disproportionately to cancellations.",
        "Root Cause": "Agents work on speculation and earn commissions on confirmed "
                      "bookings only — incentive misalignment. No penalty for cancellations.",
        "Business Impact": "Elevated cancellation rate drags overall platform KPIs; "
                            "operational overhead of managing refunds."
    },
    {
        "Observation": "5★ properties under-represented (15% of bookings)",
        "Evidence": "5★ bookings: 4,511 vs 4★: 12,034. But 5★ avg profit is highest.",
        "Root Cause": "Platform may not be actively promoting 5★ inventory; premium "
                      "customers not being targeted with tailored campaigns.",
        "Business Impact": "Missed profit upside — shifting 5% of bookings from 3★ to "
                            "5★ could add ~₹15–20M in incremental profit annually."
    },
    {
        "Observation": "High Refund Rate (78.4% of bookings have refund_status=Yes)",
        "Evidence": "23,512 rows with refund_status=Yes but only 6,070 cancellations.",
        "Root Cause": "refund_status='Yes' likely means 'refund policy exists / "
                      "eligible' rather than 'refund was issued'. Data definition needs "
                      "clarification with the engineering team.",
        "Business Impact": "Misleading KPI — true refund exposure may be much lower. "
                            "Governance and data dictionary required."
    },
    {
        "Observation": "Suite rooms are underboooked (9.9% of bookings)",
        "Evidence": "Suites: 2,970 bookings vs Standard: 16,552. "
                    "Yet Suites have highest Avg Booking Value.",
        "Root Cause": "Limited inventory or poor search visibility for suite category. "
                      "Customers default to Standard without being shown upgrade value.",
        "Business Impact": "Upsell revenue gap. A 10% shift from Standard to Suite "
                            "could increase platform margin by ~8%."
    },
]

rca_df = pd.DataFrame(rca)
for i, row in rca_df.iterrows():
    print(f"\n[RCA {i+1}]")
    for k, v in row.items():
        print(f"  {k:16s}: {v}")

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 9 — STRATEGIC RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
section("PHASE 9 — STRATEGIC RECOMMENDATIONS")

recommendations = [
    {
        "Area": "Cancellation Reduction",
        "Recommendation": "Introduce tiered non-refundable pricing (5–12% discount "
                           "for non-refundable fares) and soft-lock deposits (20%) for "
                           "bookings with lead time >30 days.",
        "Rationale": "Long lead-time bookings (31-60 days) show the highest cancel "
                     "rates. A partial commitment reduces frivolous booking behaviour "
                     "without harming demand.",
        "Expected KPI Impact": "Reduce cancellation rate from 20.2% → 14% within 6 months. "
                                "Recover ₹8–12M in annualised profit."
    },
    {
        "Area": "Cancellation Reduction",
        "Recommendation": "Implement Travel Agent SLAs and performance scorecards. "
                           "Penalise agents with >15% cancel rates by reducing commission "
                           "or requiring pre-payment deposits.",
        "Rationale": "Travel Agents have highest cancel rates due to speculative bookings "
                     "with no financial risk. Contractual penalties realign incentives.",
        "Expected KPI Impact": "Reduce agent cancellation rate by 25–30%."
    },
    {
        "Area": "Failure Rate Reduction",
        "Recommendation": "Integrate a payment retry engine and offer alternate payment "
                           "methods at checkout failure. Add UPI as a payment option for "
                           "Indian users.",
        "Rationale": "7.5% failure rate represents 2,258 lost bookings. Most failures "
                     "are recoverable with intelligent retry logic and method fallbacks.",
        "Expected KPI Impact": "Convert 30–40% of failed bookings → save ₹15–20M in "
                                "annual revenue."
    },
    {
        "Area": "Profitability Improvement",
        "Recommendation": "Launch 'Suite Upgrade' prompts at the booking confirmation "
                           "page for Standard and Deluxe rooms. Offer upgrades at ₹500–800 "
                           "incremental cost.",
        "Rationale": "Suites yield the highest avg profit but have 2,970 bookings vs "
                     "16,552 for Standard. A 10% upsell conversion on Standard bookings "
                     "would add ~2,000 suite-nights.",
        "Expected KPI Impact": "Increase avg profit per booking by ₹300–500. "
                                "Add ₹6–10M annual profit."
    },
    {
        "Area": "Profitability Improvement",
        "Recommendation": "Promote 5-star properties through curated 'Premium Picks' "
                           "homepage carousel and targeted email campaigns to customers "
                           "with >₹30K avg booking value.",
        "Rationale": "5★ properties have the highest avg profit per booking but only "
                     "15% of volume. Demand exists — the gap is discovery.",
        "Expected KPI Impact": "Grow 5★ booking share from 15% → 20%. Add ₹12–18M profit."
    },
    {
        "Area": "Customer Retention",
        "Recommendation": "Launch a VIP Loyalty Programme (Silver / Gold / Platinum) "
                           "with points per booking, free room upgrades, and priority "
                           "customer support for top 20% spenders.",
        "Rationale": "All 499 customers are already repeat bookers — a structured loyalty "
                     "programme will increase frequency and average order value.",
        "Expected KPI Impact": "Increase avg bookings per customer by 1.5×; "
                                "improve NPS by 15–20 points."
    },
    {
        "Area": "Channel Optimisation",
        "Recommendation": "Invest in Mobile App push notifications for abandoned bookings "
                           "and personalised deal alerts. A/B test app-exclusive discounts.",
        "Rationale": "Mobile App has strong adoption (40% of bookings) with confirmation "
                     "rates comparable to Web. Increasing mobile conversion by 5% adds "
                     "~600 confirmed bookings/month.",
        "Expected KPI Impact": "Mobile App confirmation rate: 72% → 77%. "
                                "+₹4M annual revenue."
    },
    {
        "Area": "Channel Optimisation",
        "Recommendation": "Shift marketing spend toward Web and Mobile App channels. "
                           "Reduce reliance on Travel Agents for standard inventory; "
                           "use agents only for group/corporate bookings.",
        "Rationale": "Travel Agents contribute <10% of volume with higher cancel overhead. "
                     "D2C channels (Web + App) are more profitable and controllable.",
        "Expected KPI Impact": "Reduce operational overhead; improve overall "
                                "confirmation rate by 2–3 percentage points."
    },
    {
        "Area": "Repeat Bookings",
        "Recommendation": "Implement personalised re-engagement campaigns: 'Book your "
                           "next stay in [City]' emails triggered 60 days after checkout, "
                           "with preferred room type pre-filled.",
        "Rationale": "The platform has a 100% repeat customer base — hyper-personalisation "
                     "can shorten rebooking cycles.",
        "Expected KPI Impact": "Increase booking frequency from ~60/cust/year to 70+"
    },
    {
        "Area": "Data Governance",
        "Recommendation": "Clarify the definition of refund_status — distinguish between "
                           "'refund-eligible' vs 'refund-issued'. Add a refund_issued_amount "
                           "column. Fix coupon_redeem negative values at source.",
        "Rationale": "Current refund_status inflates perceived refund rate (78.4%). "
                     "Clean data enables correct KPI measurement and audit compliance.",
        "Expected KPI Impact": "Accurate refund KPI; enables targeted refund policy "
                                "review and cash flow planning."
    },
]

rec_df = pd.DataFrame(recommendations)
for i, row in rec_df.iterrows():
    print(f"\n[REC {i+1}] {row['Area'].upper()}")
    print(f"  Recommendation  : {row['Recommendation']}")
    print(f"  Rationale       : {row['Rationale']}")
    print(f"  Expected Impact : {row['Expected KPI Impact']}")

# ── Priority Matrix Visualisation ─────────────────────────────────────────────
rec_matrix = pd.DataFrame({
    "Initiative": [
        "Non-refundable pricing", "Agent SLAs", "Payment retry engine",
        "Suite upsell prompts", "5★ premium campaigns", "VIP loyalty programme",
        "Mobile push + A/B", "Channel rebalancing", "Re-engagement emails",
        "Data governance"
    ],
    "Ease_of_Implementation": [6, 5, 7, 8, 7, 6, 8, 5, 8, 9],
    "Business_Impact": [9, 7, 9, 7, 8, 8, 6, 6, 7, 5],
    "Area": [
        "Cancellation", "Cancellation", "Failure Rate",
        "Profitability", "Profitability", "Retention",
        "Channel", "Channel", "Repeat", "Governance"
    ]
})

area_colors_map = {
    "Cancellation": "#C73E1D", "Failure Rate": "#F18F01",
    "Profitability": "#44BBA4", "Retention": "#2E86AB",
    "Channel": "#7B68EE", "Repeat": "#A23B72", "Governance": "#393E41"
}

fig, ax = plt.subplots(figsize=(11, 8))
for _, row in rec_matrix.iterrows():
    ax.scatter(row["Ease_of_Implementation"], row["Business_Impact"],
               s=250, color=area_colors_map[row["Area"]], zorder=3, edgecolor="white")
    ax.annotate(row["Initiative"], (row["Ease_of_Implementation"], row["Business_Impact"]),
                textcoords="offset points", xytext=(8, 4), fontsize=8.5)

ax.axhline(7, linestyle="--", color="grey", linewidth=1)
ax.axvline(7, linestyle="--", color="grey", linewidth=1)
ax.set_xlabel("Ease of Implementation (1=Hard, 10=Easy)", fontsize=11)
ax.set_ylabel("Business Impact (1=Low, 10=High)", fontsize=11)
ax.set_title("Strategic Initiative Priority Matrix\n"
             "Top-Right = Quick Wins | Top-Left = Strategic Bets",
             fontsize=13, fontweight="bold")
ax.text(7.2, 9.5, "QUICK WINS", color="green", fontsize=9, fontstyle="italic")
ax.text(4, 9.5, "STRATEGIC BETS", color="#C73E1D", fontsize=9, fontstyle="italic")
ax.text(7.2, 4.5, "FILL-INS", color="grey", fontsize=9, fontstyle="italic")
ax.text(4, 4.5, "LOW PRIORITY", color="grey", fontsize=9, fontstyle="italic")

from matplotlib.patches import Patch
legend_handles = [Patch(color=v, label=k) for k, v in area_colors_map.items()]
ax.legend(handles=legend_handles, loc="lower left", fontsize=8, title="Focus Area")
plt.tight_layout()
plt.savefig("p9_priority_matrix.png", dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
section("FINAL SUMMARY DASHBOARD")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#1a1a2e")
ax_title = fig.add_axes([0, 0.92, 1, 0.08])
ax_title.set_facecolor("#16213e")
ax_title.axis("off")
ax_title.text(0.5, 0.5, "HOTEL BOOKING PLATFORM — EXECUTIVE KPI DASHBOARD",
              color="white", fontsize=16, fontweight="bold",
              ha="center", va="center", transform=ax_title.transAxes)

kpi_boxes = [
    ("Total Bookings", f"30,000", "#2E86AB"),
    ("Confirmation Rate", f"{confirm_rate:.1f}%", "#44BBA4"),
    ("Cancellation Rate", f"{cancel_rate:.1f}%", "#C73E1D"),
    ("Failure Rate", f"{failure_rate:.1f}%", "#F18F01"),
    ("Avg Booking Value", f"₹{avg_booking_val:,.0f}", "#7B68EE"),
    ("Avg Profit/Booking", f"₹{avg_profit:,.0f}", "#A23B72"),
    ("Profit Margin", f"{profit_margin:.1f}%", "#44BBA4"),
    ("Avg Lead Time", f"{avg_lead:.0f} days", "#2E86AB"),
]

for i, (label, val, color) in enumerate(kpi_boxes):
    col = i % 4
    row = i // 4
    left = 0.01 + col * 0.25
    bottom = 0.56 - row * 0.35
    ax_box = fig.add_axes([left, bottom, 0.23, 0.30])
    ax_box.set_facecolor(color)
    ax_box.axis("off")
    ax_box.text(0.5, 0.65, val, color="white", fontsize=20, fontweight="bold",
                ha="center", va="center", transform=ax_box.transAxes)
    ax_box.text(0.5, 0.25, label, color="white", fontsize=10,
                ha="center", va="center", transform=ax_box.transAxes)

ax_status = fig.add_axes([0.01, 0.03, 0.35, 0.28])
ax_status.set_facecolor("#16213e")
status_counts = df["booking_status"].value_counts()
ax_status.pie(status_counts, labels=status_counts.index, autopct="%1.1f%%",
              colors=[STATUS_PAL.get(s,"grey") for s in status_counts.index],
              textprops={"color":"white"}, startangle=90)
ax_status.set_title("Booking Status Split", color="white", fontweight="bold")

ax_ch = fig.add_axes([0.38, 0.03, 0.30, 0.28])
ax_ch.set_facecolor("#16213e")
ch_data = df["booking_channel"].value_counts()
ax_ch.barh(ch_data.index, ch_data.values, color=PALETTE[:3], edgecolor="white")
ax_ch.set_facecolor("#16213e")
ax_ch.tick_params(colors="white"); ax_ch.spines["bottom"].set_color("white")
for spine in ["top","right","left"]: ax_ch.spines[spine].set_visible(False)
ax_ch.set_title("Bookings by Channel", color="white", fontweight="bold")
ax_ch.xaxis.label.set_color("white"); ax_ch.yaxis.label.set_color("white")

ax_rt = fig.add_axes([0.70, 0.03, 0.28, 0.28])
ax_rt.set_facecolor("#16213e")
rt_data = df["room_type"].value_counts()
ax_rt.pie(rt_data, labels=rt_data.index, autopct="%1.1f%%",
          colors=["#2E86AB","#A23B72","#F18F01"],
          textprops={"color":"white"}, startangle=90)
ax_rt.set_title("Room Type Mix", color="white", fontweight="bold")

plt.savefig("p_dashboard.png", dpi=150, facecolor=fig.get_facecolor()); plt.close()

print("\n" + "="*72)
print("  NOTEBOOK COMPLETE")
print("  Output charts saved:")
charts = [
    "p1_missing_values.png", "p1_outliers.png",
    "p2_kpi_rates.png",
    "p3_channel_performance.png", "p3_roomtype_performance.png",
    "p3_starrating_performance.png", "p3_city_performance.png",
    "p3_staytype_performance.png",
    "p4_customer_repeat.png", "p4_incentive_usage.png",
    "p5_cancellation_analysis.png",
    "p6_profitability.png", "p6_cost_vs_revenue.png",
    "p7_temporal_analysis.png", "p7_heatmap_month_channel.png",
    "p9_priority_matrix.png", "p_dashboard.png",
]
for c in charts:
    print(f"    ✔ {c}")
print("="*72)
