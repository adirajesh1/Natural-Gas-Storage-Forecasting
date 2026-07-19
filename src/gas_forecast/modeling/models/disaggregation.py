from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from gas_forecast.data.balance_api import NATIONAL_BASELINE_SERIES, STATE_TO_ABBR


class StructuralDisaggregator:
    """
    Fits structural models on monthly EIA state-level natural gas data
    and disaggregates them to weekly values using weather and price covariates.
    """
    def __init__(self):
        self.res_com_model = LinearRegression()
        self.power_burn_model = LinearRegression()
        self.monthly_means = {}  # Fallbacks for calendar month profiles

    def fit(
        self,
        monthly_df: pd.DataFrame,
        daily_weather: pd.DataFrame,
        daily_price: pd.DataFrame,
        states: list[str],
    ):
        """
        Fit weather and price regressions on monthly data.
        """
        # 1. Extract national series and pivot
        national_series = list(NATIONAL_BASELINE_SERIES)
        available_series = set(monthly_df["series"].dropna().astype(str))
        missing_national = sorted(set(national_series) - available_series)
        if missing_national:
            raise ValueError(
                "Monthly EIA data is missing required national baseline series: "
                f"{missing_national}. Refresh the balance cache and try again."
            )
        national_df = monthly_df[monthly_df["series"].isin(national_series)].copy()
        national_df["value_bcf"] = national_df["value"] / 1000.0
        national_monthly = national_df.pivot(index="period", columns="series", values="value_bcf").reset_index()
        for col in national_series:
            if col not in national_monthly.columns:
                national_monthly[col] = 0.0

        # Filter for state-level series
        state_df = monthly_df[~monthly_df["series"].isin(national_series)].copy()
        
        def map_series_to_component(series: str) -> str | None:
            if series.startswith("N3010"):
                return "res"
            elif series.startswith("N3020"):
                return "com"
            elif series.startswith("N3035"):
                return "ind"
            elif series.startswith("N3045"):
                return "power_burn"
            elif series.startswith("N3050"):
                return "price"
            elif series.startswith("N9050"):
                return "marketed_production"
            elif series.startswith("NA1160_S"):
                return "dry_production"
            return None

        state_df["component"] = state_df["series"].apply(map_series_to_component)
        state_df = state_df.dropna(subset=["component"])
        
        # Legacy marketed-production IDs embed the state at positions 5:7;
        # current dry-production IDs use the ``_SXX_`` form.
        state_df["abbr"] = state_df["series"].str.slice(5, 7)
        dry_production = state_df["component"].eq("dry_production")
        state_df.loc[dry_production, "abbr"] = state_df.loc[
            dry_production, "series"
        ].str.extract(r"_S([A-Z]{2})_", expand=False)
        
        # Filter explicitly to the region's states to avoid double-counting national/other totals
        state_abbrs = {STATE_TO_ABBR.get(s) for s in states if STATE_TO_ABBR.get(s)}
        state_df = state_df[state_df["abbr"].isin(state_abbrs)]

        # Deduplicate and pivot to columns
        state_df = state_df.drop_duplicates(subset=["period", "abbr", "component"])
        pivoted = state_df.pivot(index=["period", "abbr"], columns="component", values="value").reset_index()
        
        # Forward-fill marketed_production per state before zeroing
        pivoted = pivoted.sort_values("period")
        production_columns = ["marketed_production", "dry_production"]
        component_columns = [
            "res", "com", "ind", "power_burn", "price", *production_columns
        ]
        for col in component_columns:
            if col not in pivoted.columns:
                pivoted[col] = np.nan
        
        # Aggregate daily prices to monthly averages once for reuse and fallback
        price_m = daily_price.copy()
        price_m["period"] = price_m["period"].dt.to_period("M").dt.to_timestamp()
        monthly_price = price_m.groupby("period")["value"].mean().rename("price_hh").reset_index()
        pivoted = pivoted.merge(monthly_price, on="period", how="left")

        for col in component_columns:
            pivoted[col] = pivoted.groupby("abbr")[col].transform(lambda s: s.ffill())
            if col == "price":
                # Fall back to Henry Hub price if a state has no price history at all
                fallback = pivoted["price_hh"] if "price_hh" in pivoted.columns else 3.0
                pivoted[col] = pivoted[col].fillna(fallback).fillna(3.0)
            else:
                pivoted[col] = pivoted[col].fillna(0.0)
                
        # Convert MMcf to Bcf for volumes
        # Some EIA aliases resolve to dry production instead of marketed
        # production. Prefer the directly reported dry value if both appear.
        pivoted.loc[
            pivoted["dry_production"] > 0,
            "marketed_production",
        ] = 0.0
        for col in ["res", "com", "ind", "power_burn", *production_columns]:
            pivoted[f"{col}_bcf"] = pivoted[col] / 1000.0
            
        # Compute retail consumption in Bcf (Res + Com + Ind + Power Burn)
        pivoted["consumption_bcf"] = (
            pivoted["res_bcf"] + pivoted["com_bcf"] + pivoted["ind_bcf"] + pivoted["power_burn_bcf"]
        )
        
        # Compute consumption-weighted price ($/Mcf, same scale as $/MMBtu)
        pivoted["price_weighted"] = pivoted["price"] * pivoted["consumption_bcf"]
        
        # Aggregate by period
        regional_monthly = pivoted.groupby("period").agg(
            res_com_sum=pd.NamedAgg(column="res_bcf", aggfunc="sum"),
            com_sum=pd.NamedAgg(column="com_bcf", aggfunc="sum"),
            ind_sum=pd.NamedAgg(column="ind_bcf", aggfunc="sum"),
            power_burn_sum=pd.NamedAgg(column="power_burn_bcf", aggfunc="sum"),
            marketed_production_sum=pd.NamedAgg(column="marketed_production_bcf", aggfunc="sum"),
            dry_production_sum=pd.NamedAgg(column="dry_production_bcf", aggfunc="sum"),
            total_price_weighted=pd.NamedAgg(column="price_weighted", aggfunc="sum"),
            total_consumption=pd.NamedAgg(column="consumption_bcf", aggfunc="sum")
        ).reset_index()
        
        regional_monthly["res_com"] = regional_monthly["res_com_sum"] + regional_monthly["com_sum"]
        regional_monthly["power_burn"] = regional_monthly["power_burn_sum"]
        regional_monthly["ind"] = regional_monthly["ind_sum"]
        regional_monthly["marketed_production"] = regional_monthly["marketed_production_sum"]
        regional_monthly["reported_dry_production"] = regional_monthly["dry_production_sum"]
        
        regional_monthly["regional_price"] = np.where(
            regional_monthly["total_consumption"] > 0,
            regional_monthly["total_price_weighted"] / regional_monthly["total_consumption"],
            0.0
        )
        
        # Merge regional aggregates with national series
        merged_monthly = regional_monthly.merge(national_monthly, on="period", how="inner")
        
        # Use reported state dry production where EIA supplies it. For states
        # represented by marketed production, apply the national dry/marketed ratio.
        dry_to_marketed_ratio = np.where(
            merged_monthly["N9050US2"] > 0,
            merged_monthly["N9070US2"] / merged_monthly["N9050US2"],
            np.nan
        )
        dry_to_marketed_ratio = pd.Series(dry_to_marketed_ratio).ffill().fillna(0.0).values
        merged_monthly["dry_production"] = (
            merged_monthly["reported_dry_production"]
            + merged_monthly["marketed_production"] * dry_to_marketed_ratio
        )
        prod_ratio = np.where(
            merged_monthly["N9070US2"] > 0,
            merged_monthly["dry_production"] / merged_monthly["N9070US2"],
            0.0,
        )
        
        # Downscale national fuel use components to regional totals
        merged_monthly["retail_consumption"] = (
            merged_monthly["res_com"] +
            merged_monthly["power_burn"] +
            merged_monthly["ind"]
        )
        
        raw_cons_ratio = np.where(
            merged_monthly["N9140US2"] > 0,
            merged_monthly["retail_consumption"] / merged_monthly["N9140US2"],
            np.nan
        )
        cons_ratio = pd.Series(raw_cons_ratio).ffill().fillna(0.0).values
        
        merged_monthly["lease_plant_fuel"] = merged_monthly["N9160US2"] * prod_ratio
        merged_monthly["pipeline_fuel"] = merged_monthly["N9170US2"] * cons_ratio
        merged_monthly["fuel_use"] = merged_monthly["lease_plant_fuel"] + merged_monthly["pipeline_fuel"]
        
        # Final regional dataset (retaining regional_price for price basis spreading)
        regional_monthly = merged_monthly[["period", "dry_production", "res_com", "power_burn", "ind", "fuel_use", "regional_price"]]
        
        # 2. Aggregate daily weather to monthly totals
        weather_m = daily_weather.copy()
        weather_m["period"] = weather_m["date"].dt.to_period("M").dt.to_timestamp()
        monthly_weather = weather_m.groupby("period")[["hdd", "cdd"]].sum().reset_index()

        # 3. Re-use aggregated monthly prices calculated earlier

        # 4. Merge monthly balance, weather, and Henry Hub price data
        merged = regional_monthly.merge(monthly_weather, on="period", how="inner")
        merged = merged.merge(monthly_price, on="period", how="inner")
        
        if merged.empty:
            def period_range(frame: pd.DataFrame) -> str:
                if frame.empty:
                    return "empty"
                return f"{frame['period'].min():%Y-%m} to {frame['period'].max():%Y-%m}"

            raise ValueError(
                "No overlapping monthly data found between EIA metrics, weather, "
                "and price. Ranges: "
                f"EIA={period_range(regional_monthly)}, "
                f"weather={period_range(monthly_weather)}, "
                f"price={period_range(monthly_price)}."
            )

        # Calculate seasonal basis spread (Regional Price - Henry Hub Price)
        merged["basis_spread"] = merged["regional_price"] - merged["price_hh"]
        merged["month"] = merged["period"].dt.month
        self.seasonal_basis_spread = merged.groupby("month")["basis_spread"].mean().to_dict()

        # Compute number of days in each calendar month for per-day scaling
        merged["days_in_month"] = merged["period"].dt.days_in_month
        
        # Per-day conversions
        merged["res_com_per_day"] = merged["res_com"] / merged["days_in_month"]
        merged["power_burn_per_day"] = merged["power_burn"] / merged["days_in_month"]
        merged["hdd_per_day"] = merged["hdd"] / merged["days_in_month"]
        merged["cdd_per_day"] = merged["cdd"] / merged["days_in_month"]
        
        # 5. Fit Regressions
        # Res/Com model
        X_rc = merged[["hdd_per_day", "cdd_per_day"]]
        y_rc = merged["res_com_per_day"]
        self.res_com_model.fit(X_rc, y_rc)
        
        # Power Burn model fit on regional price
        X_pb = merged[["hdd_per_day", "cdd_per_day", "regional_price"]]
        y_pb = merged["power_burn_per_day"]
        self.power_burn_model.fit(X_pb, y_pb)

        # 6. Fit profile averages of daily rates for industrial, production, and fuel use
        self.seasonal_profiles = {}
        for col in ["dry_production", "ind", "fuel_use"]:
            daily_rate_col = f"{col}_per_day"
            merged[daily_rate_col] = merged[col] / merged["days_in_month"]
            self.seasonal_profiles[col] = merged.groupby("month")[daily_rate_col].mean().to_dict()
        
        # Keep the full merged dataframe for interpolation references
        self.monthly_history = merged.sort_values("period").reset_index(drop=True)

    def predict_weekly(
        self,
        weekly_weather: pd.DataFrame,
        weekly_price: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Estimate weekly components using fitted regressions and interpolations.
        """
        # Align weather and price on Friday dates
        weekly = weekly_weather.copy()
        weekly = weekly.merge(weekly_price[["period", "value"]].rename(columns={"period": "date", "value": "price"}), on="date", how="inner")
        
        if weekly.empty:
            return pd.DataFrame()

        # Compute weekly weather rates (per day)
        weekly["hdd_per_day"] = weekly["hdd"] / 7.0
        weekly["cdd_per_day"] = weekly["cdd"] / 7.0

        # Predict Res/Com (Bcf/day) and scale to 7 days
        X_rc = weekly[["hdd_per_day", "cdd_per_day"]]
        weekly["res_com"] = self.res_com_model.predict(X_rc) * 7.0
        # Prevent negative consumption
        weekly["res_com"] = weekly["res_com"].clip(lower=0.0)

        # Compute weekly regional price using the basis spread model
        weekly["month_num"] = weekly["date"].dt.month
        weekly["basis_spread"] = weekly["month_num"].map(self.seasonal_basis_spread).fillna(0.0)
        weekly["regional_price"] = weekly["price"] + weekly["basis_spread"]

        # Predict Power Burn (Bcf/day) using regional price, then scale to 7 days
        X_pb = weekly[["hdd_per_day", "cdd_per_day", "regional_price"]]
        weekly["power_burn"] = self.power_burn_model.predict(X_pb) * 7.0
        weekly["power_burn"] = weekly["power_burn"].clip(lower=0.0)

        # Interpolate/project Production, Industrial, and Fuel Use
        hist = self.monthly_history.copy()
        
        # Center dates for monthly data (e.g. 15th of the month)
        hist["center_date"] = hist["period"] + pd.to_timedelta(hist["days_in_month"] / 2.0, unit="D")
        max_hist_center = hist["center_date"].max()
        
        interpolated = {}
        for col in ["dry_production", "ind", "fuel_use"]:
            daily_rate_col = f"{col}_per_day"
            hist[daily_rate_col] = hist[col] / hist["days_in_month"]
            
            # Calculate scaling factor based on the last available month's actual value vs its seasonal average
            last_row = hist.iloc[-1]
            last_month = last_row["period"].month
            last_actual_rate = last_row[daily_rate_col]
            month_profile = self.seasonal_profiles[col]
            last_seasonal_rate = month_profile.get(last_month, np.mean(list(month_profile.values())) if month_profile else 0.0)
            
            if last_seasonal_rate > 0 and last_actual_rate > 0:
                ratio = np.clip(last_actual_rate / last_seasonal_rate, 0.5, 1.5)
            else:
                ratio = 1.0
            
            col_predictions = []
            for date in weekly["date"]:
                if date <= max_hist_center:
                    # Interpolate within historical range
                    val = np.interp(
                        [date.value],
                        hist["center_date"].astype(np.int64),
                        hist[daily_rate_col]
                    )[0]
                else:
                    # Project forward using the ratio-to-seasonal profile extrapolation
                    t_month = date.month
                    default_rate = np.mean(list(month_profile.values())) if month_profile else 0.0
                    val = month_profile.get(t_month, default_rate) * ratio
                col_predictions.append(val * 7.0)
                
            interpolated[col] = col_predictions

        weekly["dry_production"] = interpolated["dry_production"]
        weekly["industrial"] = interpolated["ind"]
        weekly["fuel_use"] = interpolated["fuel_use"]

        # Calculate local supply-demand balance (dry production - consumption)
        # Note: regional net balance = Production - (ResCom + PowerBurn + Industrial + Fuel Use)
        # Positive means supply surplus (likely injecting into storage or exporting),
        # Negative means supply deficit (withdrawing from storage or importing).
        weekly["local_balance"] = (
            weekly["dry_production"] -
            (weekly["res_com"] + weekly["power_burn"] + weekly["industrial"] + weekly["fuel_use"])
        )

        return weekly
