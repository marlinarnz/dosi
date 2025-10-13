import pandas as pd

ONEDRIVE_PATH = (
    "/mnt/c/Users/simon.destercke/IIASA/EDITS - FT25-1_PosTip/Data/HATCH files/"
)
fn_hatch_data = ONEDRIVE_PATH + "all_tech_clean_cumulative_Update.csv"
fn_hatch_early = ONEDRIVE_PATH + "combined_hatch_characteristics_earlylate.xlsx"
fn_hatch_fits = ONEDRIVE_PATH + "hatch_countrylevel_AllGrowthMerge.csv"
fn_hatch_clusters = (
    ONEDRIVE_PATH + "../Copy of summary_table_v26_CW_28May_Cornwall.xlsx"
)
sn_hatch_clusters = "CW analysis0_23SIs+CLUSTERFUNC"


def get_early_dict(fn):
    df_early = pd.read_excel(fn)
    df_early = df_early[df_early["Country_Timing"] == "Early"]
    early_dict = dict(zip(df_early["technology"], df_early["country"]))
    return early_dict


def get_hatch_clusters_dict(fn, sn):
    hatch_clusters_df = pd.read_excel(
        fn,
        sheet_name=sn,
        skiprows=45,
        usecols=[35, 36, 37, 38, 39],
        # encoding="ISO-8859-1",
        header=0,
    )

    hatch_clusters_dict = {
        col: hatch_clusters_df[col]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.strip() != ""]
        .tolist()
        for col in hatch_clusters_df.columns
    }

    return hatch_clusters_dict


def load_hatch_data(fn, filter_regions=None, filter_technologies=None):
    df = pd.read_csv(fn, index_col=0)
    if filter_regions is not None:
        df = df[df["Country Code"].isin(filter_regions)]
    if filter_technologies is not None:
        df = df[df["Technology Name"].isin(filter_technologies)]
    # Melt, keeping all columns that are not numbers, but without hardcoding them
    df = df.melt(
        id_vars=[
            df.columns[i] for i in range(len(df.columns)) if not df.columns[i].isdigit()
        ],
        var_name="Year",
        value_name="Value",
    )
    # remove rows with NA in Value column
    df = df.dropna(subset=["Value"])
    # Convert Year to integer
    df["Year"] = df["Year"].astype(int)
    # Convert Value to float
    df["Value"] = df["Value"].astype(float)

    print(df["Technology Name"].unique())
    return df


def hatch_to_dosi_format(df):
    df = df.rename(
        columns={
            "Technology Name": "Innovation Name",
            "Metric": "Description",
            "Unit": "Metric",
        }
    )
    hatch_mapping = pd.read_csv("hatch_mapping.csv")
    # Strip leading and trailing spaces in all columns of hatch_mapping
    hatch_mapping = hatch_mapping.map(lambda x: x.strip() if isinstance(x, str) else x)
    print(hatch_mapping)
    # Merge df with hatch_mapping on Spatial Scale
    df = df.merge(
        hatch_mapping,
        how="left",
        left_on="Country Code",
        right_on="Early dict value (ISO3)",
    )
    df["Spatial Scale"] = df["Full name"]
    df["Innovation Name"] = df["Innovation Name"].str.lower()
    df["Indicator Number"] = "1.1"
    df["Indicator Name"] = "Adoption over Time"
    df["Comments"] = "HATCH data"
    df["File"] = "HATCH files"
    df["Sheet"] = "HATCH"
    df = df[
        [
            "Year",
            "Value",
            "Innovation Name",
            "Indicator Number",
            "Indicator Name",
            "Description",
            "Metric",
            "Data Source",
            "Comments",
            "Spatial Scale",
            "File",
            "Sheet",
        ]
    ]
    return df


# HARD CODE, with ChatGPT:

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


if __name__ == "__main__":
    early_dict = get_early_dict(fn_hatch_early)
    hatch_clusters_dict = get_hatch_clusters_dict(fn_hatch_clusters, sn_hatch_clusters)

    # Extract all the technologies from the clusters, taking the unique values of the list concatenation of all the keys
    technologies_in_clusters = list(
        set(tech for techs in hatch_clusters.values() for tech in techs)
    )
    # Add technologies that are in technologies_in_clusters but not in early dict keys, to early_dict with value "USA"
    for tech in technologies_in_clusters:
        if tech not in early_dict:
            early_dict[tech] = "USA"

    # Get the list of early adopting regions for the technologies in the clusters, taking the unique values
    early_adopters = list(
        set(early_dict[tech] for tech in technologies_in_clusters if tech in early_dict)
    )

    # Outputs
    # print("Technologies:", technologies_in_clusters)
    # print("Early adopting regions:", early_adopters)

    # print("Early dict:", early_dict)
    # print("Hatch clusters dict:", hatch_clusters_dict)

    hatch_data_df = load_hatch_data(
        fn_hatch_data,
        filter_regions=early_adopters,
        filter_technologies=technologies_in_clusters,
    )
    hatch_to_dosi_format(hatch_data_df).to_csv(
        ONEDRIVE_PATH + "/hatch_data_dosi_format.csv", index=False
    )

    print("Hatch data df:", hatch_data_df.head())

    # Example usage:
    # print(early_dict)
    # print(hatch_clusters_dict)
