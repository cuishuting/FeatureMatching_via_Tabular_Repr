import pandas as pd
import numpy as np


# Only lab and Chart_Num itemids needed to be imputed
def Imputation(fill_limit_list, org_itemids_tabular, era, dataset, recording_range=24):
    # param "org_itemids_tabular" is the result from Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular
    # param "era" == "CV" or "MV"
    # param "dataset" == "lab" or "chart"

    # Firstly extend current pd DataFrame with 24 rows


    if len(org_itemids_tabular["time_window"]) != recording_range:
        current_time_window = set(org_itemids_tabular["time_window"])
        expected_time_window = set(range(1, recording_range + 1))
        missed_time_window = expected_time_window - current_time_window
        # "df_interpolate_rows" is the rows being interpolated on org_itemids_tabular with all NaN values
        df_interpolate_rows = pd.DataFrame(np.nan, index=missed_time_window, columns=org_itemids_tabular.columns[1:])
        df_interpolate_rows["time_window"] = df_interpolate_rows.index
        first_col = df_interpolate_rows.pop("time_window")
        df_interpolate_rows.insert(0, "time_window", first_col)
        imputed_itemid_recordings = pd.concat([org_itemids_tabular, df_interpolate_rows]).sort_values(by="time_window")
        imputed_itemid_recordings.reset_index(inplace=True, drop=True)
    else:
        imputed_itemid_recordings = org_itemids_tabular

    cur_hadm_itemids = org_itemids_tabular.columns[1:]
    # print(cur_hadm_itemids)
    for itemid in cur_hadm_itemids:
        if era == "CV":
            # print(fill_limit_list["itemid"].unique())
            # print(fill_limit_list[fill_limit_list["itemid"] == itemid])
            cur_itemid_fill_limit = round(fill_limit_list[fill_limit_list["itemid"] == int(itemid)]["recommend_fill_limit_CV"].values[0])
        else:  # era == "MV"
            cur_itemid_fill_limit = round(fill_limit_list[fill_limit_list["itemid"] == int(itemid)]["recommend_fill_limit_MV"].values[0])
        if dataset == "lab":
            imputed_itemid_recordings[itemid] = pd.to_numeric(imputed_itemid_recordings[itemid], errors='coerce')

        cur_itemid_recordings = imputed_itemid_recordings[[itemid]]
        imputed_cur_itemid_recordings = cur_itemid_recordings.interpolate(method="linear", axis=0, limit=cur_itemid_fill_limit, limit_area="inside")
        imputed_cur_itemid_recordings = imputed_cur_itemid_recordings.interpolate(axis=0, limit=cur_itemid_fill_limit, limit_direction="both", limit_area="outside")
        imputed_itemid_recordings[[itemid]] = imputed_cur_itemid_recordings

    return imputed_itemid_recordings
