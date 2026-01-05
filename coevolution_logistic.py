# Script from Marlin to find co-evolutions in logistic fits

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.stats import linregress


VERSION = "v27"
VERSION_FOR_FITS = "v26"
VERSION_FOR_SUMMARY_READING = "v25"
VERSION_FOR_METADATA = "v25_withhatch_2"
VERSION_FOR_DATA = "v26"
SMALL_SUBSET = False  # Do you only want a small subset for testing?
REDO_FITS = True
RENUMBER_METADATA_CODES = False
CREATE_PDFS = True
LINE_COLOR_LOG = "blue"
PATH = "./data"

def get_dosi_data():
    ###########################################
    # Code from Simon to read data and metadata
    ###########################################

    # Load DoSI
    fn_data = f"{PATH}/adjusted_datasets_{VERSION_FOR_DATA}.csv"
    adoptions_df = pd.read_csv(fn_data, converters={"Indicator Number": str})
    adoptions_df["Value"] = pd.to_numeric(adoptions_df["Value"], errors="coerce")
    adoptions_df = adoptions_df.dropna(subset=["Value"])
    # Correct for trailing spaces in the data
    adoptions_df["Spatial Scale"] = adoptions_df["Spatial Scale"].str.rstrip()
    adoptions_df["Innovation Name"] = adoptions_df["Innovation Name"].str.rstrip()
    
    # Load HATCH
    fn_data = f"{PATH}/hatch_data_dosi_format.csv"
    hatch = pd.read_csv(fn_data, converters={"Indicator Number": str})
    hatch["Value"] = pd.to_numeric(hatch["Value"], errors="coerce")
    hatch = hatch.dropna(subset=["Value"])
    hatch["Spatial Scale"] = hatch["Spatial Scale"].str.rstrip()
    hatch["Innovation Name"] = hatch["Innovation Name"].str.rstrip()

    # For debugging: only subset
    if SMALL_SUBSET:
        adoptions_df = adoptions_df[
            # adoptions_df["Innovation Name"].isin(["car sharing"])
            # adoptions_df["Indicator Number"].isin(["3.3", "3.5", "4.1"])
            # adoptions_df["Indicator Number"].isin(["1.1"])
            adoptions_df["Metric"].isin(["market share"])
        ]

    fn_metadata = f"{PATH}/metadata_master_{VERSION_FOR_METADATA}.xlsx"
    def convert_to_three_digit_notation(s):
        return re.sub(r"([a-zA-Z])(\d+)", lambda m: f"{m.group(1)}{int(m.group(2)):03}", s)

    def read_metadata_table(fn, columns):
        df = pd.read_excel(fn, usecols=columns, dtype=str).dropna().reset_index(drop=True)
        df.iloc[:, 1] = df.iloc[:, 1].apply(convert_to_three_digit_notation)
        return df.set_index(df.columns[0])[df.columns[1]].to_dict()

    categories_df = (
        pd.read_excel(fn_metadata, usecols="A,B", dtype=str).dropna().reset_index(drop=True)
    )
    categories = {
        (k.lower() if isinstance(k, str) else k): v
        for k, v in categories_df.set_index(categories_df.columns[0])[
            categories_df.columns[1]
        ].items()
    }

    metadata = dict()
    metadata["Innovation Name"] = read_metadata_table(fn_metadata, "A,D")
    metadata["Spatial Scale"] = read_metadata_table(fn_metadata, "G,I")
    metadata["Indicator Number"] = read_metadata_table(
        fn_metadata, "L,O"
    )  # Column M is the indicator name. Superfluous because maps 1-1 on indicator number
    metadata["Description"] = read_metadata_table(fn_metadata, "R,S")
    metadata["Metric"] = read_metadata_table(fn_metadata, "V,W")
    # If metadata file is not in sync with the data table, remake the description dictionary
    if RENUMBER_METADATA_CODES:
        metadata["Description"] = {
            val: convert_to_three_digit_notation(f"d{idx}")
            for idx, val in enumerate(sorted(adoptions_df["Description"].unique()), start=1)
        }
        # If metadata file is not in sync with the data table, remake the metric dictionary
        metadata["Metric"] = {
            val: convert_to_three_digit_notation(f"m{idx}")
            for idx, val in enumerate(sorted(adoptions_df["Metric"].unique()), start=1)
        }
    _, last_value = next(reversed(metadata["Description"].items()))
    description_counter = int(last_value[1:])  # Presumes a format "d023"
    _, last_value = next(reversed(metadata["Metric"].items()))
    metric_counter = int(last_value[1:])  # Presumes a format "m023"

    for key, nested_dict in metadata.items():
        if isinstance(nested_dict, dict):  # Ensure the value is a dictionary
            metadata[key] = {
                k.lower() if isinstance(k, str) else k: v for k, v in nested_dict.items()
            }
    group_vars = list(metadata.keys())

    # Load cluster assignment
    hatch_clusters = {
        "sufficiency": [
            "Electric Bicycles",  # Ebikes
            "Solar Photovoltaic",  # solar PV
        ],
        "digital": [
            "Cellphones",  # Cellphones
            "Home Computers",  # home computer
            "Household Internet Access",  # internet access
            "Microcomputers",  # microcomputers
            "Podcasting",  # podcasting
            "Real-Time Gross Settlement Adoption",  # realtime gross settlement
            "Social Media Usage",  # social media usage
        ],
        "consume": [
            "Cable TV",  # Cable TV
            "Dishwashers",  # Dishwashers
            "Electric Bicycles",  # Ebikes
            "Home Air Conditioning",  # home AC
            "Laundry Dryers",  # laundry dryers
            "Microwaves",  # microwave oven
            "Television",  # TV
            "Washing Machines",  # wash machines
        ],
        "green growth": [
            "Nox Pollution Controls (Boilers)",  # NOx pollution control
            "Offshore Wind Energy",  # offshore wind
            "Onshore Wind Energy",  # onshore wind
            "Solar Photovoltaic",  # solar PV
            "Wet Flue Gas Desulfurization Systems",  # FGD
        ],
        "health": ["Electric Bicycles"],  # Ebikes (as given)
    }
    clusters_df = pd.read_csv(
        f"{PATH}/PosTip_Clusters.csv",  # Summary file by Charlie
        skiprows=15,
        nrows=28,
        usecols=[8, 35, 36, 37, 38, 39],
        encoding="ISO-8859-1",
        header=0,
    )
    clusters_df.rename(
        columns={clusters_df.columns[0]: "innovation code"}, inplace=True
    )  # If there is an error here, then there may be a column reference error, e.g. the first column of the csv file is empty and pd.red_csv skips it
    # remove first row of clusters_df
    clusters_df.drop(index=clusters_df.index[0], axis=0, inplace=True)
    clusters_dict = {
        col: clusters_df.loc[~clusters_df[col].isna(), "innovation code"].tolist()
        for col in clusters_df.columns[1:]
    }
    for cluster, technologies in hatch_clusters.items():
        for technology in technologies:
            clusters_dict[cluster].append(metadata["Innovation Name"][technology.lower()])
    
    return adoptions_df, hatch, metadata, clusters_dict, categories


def FPLogValue_with_scaling(x, t0, Dt, K):
    """
    Logistic function with vertical scaling.
    """
    if np.isnan(t0) or np.isnan(Dt) or np.isnan(K):  # If predictions are NaN, return NaN
        return np.nan
    return K / (1 + np.exp(-np.log(81) * (x - t0) / Dt))


def logistic_3p(x, A, M, T):
    return A / (1.0 + np.exp(-M * (x - T)))
def fit_logistic_3p(x_data, y_data):
    A_guess = np.max(y_data)
    half_guess = A_guess / 2.0

    try:
        idx_half = (np.abs(y_data - half_guess)).argmin()
        T_guess = x_data[idx_half] if idx_half < len(x_data) else np.median(x_data)
    except:
        T_guess = np.median(x_data)
    M_guess = 1.0

    p0 = [A_guess, M_guess, T_guess]

    try:
        popt, pcov = curve_fit(logistic_3p, x_data, y_data, p0=p0, maxfev=10000)
        y_pred = logistic_3p(x_data, *popt)
        rmse = np.sqrt(np.mean((y_pred - y_data) ** 2))
    except RuntimeError:
        popt, pcov = np.nan, np.nan
        rmse = np.inf

    return popt, pcov, rmse


def FPLogFit_with_scaling(x, y, Dt_initial_guess: float = 10):
    """
    Fit a logistic function with vertical scaling to the data.

    Parameters:
    x (array-like): Independent variable.
    y (array-like): Dependent variable.

    Returns:
    dict: Fitted parameters t0, Dt, and S.
    """

    if len(x) < 3:
        return {"t0": np.nan, "Dt": np.nan, "K": np.nan}  # Default parameters
    else:
        popt_log, pcov_log, rmse_log = fit_logistic_3p(x, y)
        if not np.any(np.isnan(popt_log)):
            A_fit, M_fit, T_fit = popt_log
            A_err, M_err, T_err = np.sqrt(np.diag(pcov_log))
        else:
            A_fit = M_fit = T_fit = A_err = M_err = T_err = np.nan

        return {"t0": T_fit, "Dt": np.log(81) / M_fit, "K": A_fit}


def calculate_adjusted_r2(y_obs, y_pred, n_params):
    # Calculate R^2 and adjusted R^2
    if np.any(np.isnan(y_pred)):  # If predictions are NaN, return NaN
        return np.nan, np.nan
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(y_obs)
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - n_params - 1))
    return r2, r2_adj


def get_scoring_on_summary():
    #################################################
    # Code from Simon
    # Include reallocations of indicator numbers that Charlie made
    #################################################
    
    # Add in previous scoring from Charlie and Greg
    scoring_on_summary = pd.read_excel(
        f"{PATH}/summary_table_v24_21Mar.xlsx",
        sheet_name="summary_table_v24_CW",
    )
    scoring_on_summary["Innovation Name"] = scoring_on_summary[
        "Innovation Name"
    ].str.lower()

    scoring_on_summary["code_without_indicator_name"] = scoring_on_summary[
        "Code"
    ].str.replace(r"(_\d+\.\d+)[A-Za-z]{3}", r"\1", regex=True)

    def reassign_indicator_numbers(
        df,  # column name
        innovation_name,
        descriptions,
        metrics,
        old_indicator_number,
        new_indicator_number,
        cn="code_without_indicator_name",
    ):
        mask = (
            (
                (descriptions == [])
                | (
                    df[cn].str.contains(
                        "|".join(
                            [metadata["Description"][key.lower()] for key in descriptions]
                        ),
                        case=False,
                        na=False,
                    )
                )
            )
            & (
                (metrics == [])
                | (
                    df[cn].str.contains(
                        "|".join([metadata["Metric"][key.lower()] for key in metrics]),
                        case=False,
                        na=False,
                    )
                )
            )
        ) & (df[cn].str.contains(innovation_name))
        df.loc[mask, cn] = df.loc[mask, cn].str.replace(
            f"_{old_indicator_number}", f"_{new_indicator_number}"
        )
        print(f"Changed {sum(mask)} codes")
        return df


    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "dri",
        [
            "% of population holding a drivers licence, by age group<=19yrs",
        ],
        [],
        "3.2",
        "1.1",
    )

    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "dri",
        [
            "% of population (residents) holding a drivers licence",
        ],
        [],
        "1.1",
        "4.3",
    )

    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "dri",
        [
            "% of 18-19yr age group holding a drivers licence, by gender",
            "% of 18-19yr age group in 2003 holding a drivers licence",
            "share of teenagers with drivers licenses",
            "% of population holding a drivers licence, by gender=female",
            "% of population holding a drivers licence, by gender=male",
            "% of population holding a drivers licence, by age group 20-24",
        ],
        [],
        "1.1",
        "3.2",
    )

    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "sol",
        [
            "% third party owned systems (income=$100k-$150k)",
            "% third party owned systems (income=$150k-$200k)",
            "% third party owned systems (income=$200k-$250k)",
            "% third party owned systems (income>$250k)",
        ],
        [],
        "1.1",
        "3.2",
    )

    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "eat",
        [
            "% poultry+pig in total meat consumption",
            "per capita beef consumption",
            "per capita other meat consumption",
            "per capita pig consumption",
            "per capita poultry consumption",
            "per capita sheep & goat consumption",
        ],
        [],
        "1.1",
        "2.5",
    )

    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "eco",
        [
            "Annual Internet retail (B2C) sales value",
            "Enterprises' total turnover from e-commerce sales (all activities - B2B, B2C, B2G)",
        ],
        [],
        "1.1",
        "2.2",
    )

    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "org",
        [
            "organic per capita consumption [€/person]",
        ],
        [],
        "1.1",
        "2.2",
    )

    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "pas",
        [
            "new passive buildings",
        ],
        ["# of new passive buildings"],
        "1.1",
        "2.5",
    )

    scoring_on_summary = reassign_indicator_numbers(
        scoring_on_summary,
        "pas",
        [
            "new passive buildings",
        ],
        ["new floorspace (m2?)"],
        "1.1",
        "2.9",
    )
    
    return scoring_on_summary


def get_spatial_hierarchy():
    h = {}
    h['United States'] = ['California', 'Connecticut', 'Massachusetts', 'New Yersey', 'Washington DC', 'US']
    h['US'] = ['California', 'Connecticut', 'Massachusetts', 'New Yersey', 'Washington DC', 'United States']
    h['China'] = ['Beijing']
    h['The Netherlands'] = ['Amsterdam']
    h['Germany'] = ['Berlin', 'Hamburg', 'Heidelberg']
    h['Denmark'] = ['Copenhagen']
    h['Sweden'] = ['Stockholm']
    h['EU'] = h['The Netherlands'] + h['Germany'] + h['Denmark'] + h['Sweden'] \
        + ['Austria', 'Belgium', 'Denmark', 'Estonia', 'Finland', 'France',
           'Hungary', 'Ireland', 'Portugal', 'Germany', 'Latvia', 'Poland',
           'Sweden', 'The Netherlands']
    h['Europe'] = h['EU'] + ['UK', 'Norway', 'Switzerland']
    new_keys = ['Asia']
    h['Asia'] = h['China'] + ['China', 'Bangladesh', 'India', 'Japan', 'South Korea', 'Sri Lanka', 'Taiwan']
    
    return h, new_keys


##########################################
# Main script
##########################################

def logistic_fitting():
    
    dosi, hatch, metadata, clusters, categories = get_dosi_data()
    spatial_hierarchy, new_spatials = get_spatial_hierarchy()
    
    # Prepare data structures
    adjustments = []
    cluster_names = []
    indicators = []
    spatials = []
    i_name = []
    j_name = []
    i_innos = []
    j_innos = []
    i_metrics = []
    j_metrics = []
    i_descriptions = []
    j_descriptions = []
    t0 = []
    Dt = []
    K = []
    R_square = []
    R_square_adj = []
    time_lag = []
    group_vars = ['Innovation Name', 'Description', 'Metric'] # defines one time series
    dosi['name'] = dosi['Spatial Scale']
    hatch['name'] = hatch['Spatial Scale']
    for i in range(len(group_vars)):
        dosi['name'] += ' - ' + dosi[group_vars[i]]
        hatch['name'] += ' - ' + hatch[group_vars[i]]
        
    data = pd.concat([dosi,  hatch]).reset_index(drop=True)
    
    # Assign cluster definitions
    # Data explodes since an innovation may occur in multiple clusters
    data['cluster'] = [['All',] for _ in range(len(data))]
    for c, innos in clusters.items():
        mask = data['Innovation Name'].map(metadata['Innovation Name']).isin(innos)
        data.loc[mask, 'cluster'] = data.loc[mask, 'cluster'].apply(lambda l: l+[c])
    data = data.explode('cluster').reset_index(drop=True)
    
    # Analyse co-evolution for specific regions, etc.
    # pairwise for each two time series
    # Adjustments
    for adjustment in ['Original data']:# ['Original data', 't0 aligned']:
        # Look at certain innovation indicators
        for indicator in ['1.1', 'All']:
            
            # For each cluster and once irrespective of the cluster
            for cluster in list(clusters.keys()) + ['All']:
                if indicator != 'All':
                    mask = (data['Indicator Number']==indicator) & (data['cluster']==cluster)
                else:
                    mask = (data['cluster']==cluster)
                
                # For each spatial scale, including the hierarchy
                options = data.loc[mask, 'Spatial Scale'].unique()
                options = list(options) + new_spatials
                for spatial in options:
                    spatial_list = [spatial]
                    if spatial in spatial_hierarchy.keys():
                        spatial_list += spatial_hierarchy[spatial]
                    
                    # Build groups = time series
                    data_ = data.loc[mask & (data['Spatial Scale'].isin(spatial_list))]
                    groups = [g[['name', 'Value', 'Year']+group_vars] for _, g
                              in data_.groupby(group_vars)]
                    if len(groups) > 1 and len(data_) > 5:
                        
                        # Estimate pairwise fits
                        print('Estimate logistic fits for {}x{} time series in {}, {} data points'.format(
                            len(groups), len(groups), spatial, len(data_)))
                        for i in range(len(groups)):
                            for j in range(len(groups)):
                                if i>j:
                                    if adjustment == 't0 aligned':
                                        delta_t = groups[i]['Year'].mean() - groups[j]['Year'].mean()
                                        groups[i]['Year'] = groups[i]['Year'] - delta_t/2
                                        groups[j]['Year'] = groups[j]['Year'] + delta_t/2
                                    group = pd.concat([groups[i], groups[j]])
                                elif i==j:
                                    group = groups[i]
                                else:
                                    continue
                                
                                group = group.sort_values(by='Year')
                                
                                if len(set(group['Year'])) > 3:
                                    
                                    slope, _, _, _, _ = linregress(group["Year"].values,
                                                                   group["Value"].values)
                                    Dt_guess = np.log(81) / slope \
                                        * (max(group["Value"]) if max(group["Value"]) > 0 else 1) \
                                        if slope != 0 else 100
                                    result = FPLogFit_with_scaling(
                                        group["Year"].values,
                                        group["Value"].values,
                                        Dt_initial_guess=Dt_guess,
                                    )
                                    t0.append(result['t0'])
                                    Dt.append(result['Dt'])
                                    K.append(result['K'])
                                    
                                    # Estimate R²
                                    y_pred = FPLogValue_with_scaling(group["Year"], **result)
                                    r2_log, r2adj_log = calculate_adjusted_r2(group["Value"], y_pred, n_params=3)
                                    R_square.append(r2_log)
                                    R_square_adj.append(r2adj_log)
                                    
                                    delta_t = groups[i]['Year'].mean() - groups[j]['Year'].mean()
                                    time_lag.append(delta_t)
                                    
                                    adjustments.append(adjustment)
                                    cluster_names.append(cluster)
                                    indicators.append(indicator)
                                    spatials.append(spatial)
                                    i_name.append(groups[i]['name'].unique()[0])
                                    j_name.append(groups[j]['name'].unique()[0])
                                    i_innos.append(groups[i]['Innovation Name'].unique()[0])
                                    j_innos.append(groups[j]['Innovation Name'].unique()[0])
                                    i_metrics.append(groups[i]['Metric'].unique()[0])
                                    j_metrics.append(groups[j]['Metric'].unique()[0])
                                    i_descriptions.append(groups[i]['Description'].unique()[0])
                                    j_descriptions.append(groups[j]['Description'].unique()[0])
                        
    results = pd.DataFrame({'adjustment':adjustments, 'cluster':cluster_names, 'indicator':indicators,
                            'spatial':spatials, 'i_ID':i_name, 'j_ID':j_name,
                            'i_innovation':i_innos, 'j_innovation':j_innos,
                            'i_metric':i_metrics, 'j_metric':j_metrics,
                            'i_description':i_descriptions, 'j_description':j_descriptions,
                            't0':t0, 'Dt':Dt, 'K':K,
                            'R_square':R_square, 'R_square_adj':R_square_adj, 'time_lag':time_lag})
    return results


if __name__ == '__main__':
    results = logistic_fitting()
    results.to_csv(f"{PATH}/results_coevolution_logistic.csv")