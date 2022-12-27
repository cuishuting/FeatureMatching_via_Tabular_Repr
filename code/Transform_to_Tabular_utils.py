import pandas as pd
import numpy as np
import datetime


"""
Func Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings, cur_icu_adm_time, era, expected_itemids_set):
    * input: 
        1) cur_adm_recordings: From original dataset groupby "hadm_id" and then get_group(hadm_id).
        Extract only the columns [["hadm_id", "itemid", "charttime", "valuenum"(lab)/"value"(chart_num)]] from this group, 
        which combines the parameter "cur_adm_recordings"
        2) cur_icu_adm_time: mimic_icu_adm[mimic_icu_adm["hadm_id"] == hadm_id]["intime"].unique()[0] (mimic_icu_adm is read from file "mimic_icuadmittime.csv")
        3) dataset_type: only allow 2 unique values: "lab" or "chart"(represent chart_only_num)
        4) expected_itemids_list: the final tabular's expected columns, keep the same in the same dataset's same era
        (lab_cv/lab_mv/chart_cv_only_num/chart_mv_only_num)
"""


def Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings, cur_icu_adm_time, dataset_type, expected_itemids_list):
    icu_adm_timestamp = datetime.datetime.timestamp(datetime.datetime.strptime(cur_icu_adm_time, "%Y-%m-%d %H:%M:%S"))
    cur_adm_recordings["charttimestamp"] = cur_adm_recordings[["charttime"]].apply((lambda x: datetime.datetime.timestamp(datetime.datetime.strptime(x["charttime"], "%Y-%m-%d %H:%M:%S"))), axis=1)
    cur_adm_recordings[["charttime_interval_after_icu_adm"]] = (cur_adm_recordings[["charttimestamp"]] - icu_adm_timestamp) / 3600
    cur_adm_recordings[["charttime_interval_after_icu_adm"]] = cur_adm_recordings[["charttime_interval_after_icu_adm"]].astype(int) + 1
    # "+1" because astype(int) will only keep the integer part of the float number
    # for example: charttime_interval_after_icu_adm == 0.2, actually this belongs to the first one-hour window,
    # so the result should be 1, not 0 (if returned directly by astype(int))
    cur_adm_recordings.drop(columns=["charttime", "charttimestamp"], inplace=True)
    # columns: "hadm_id", "itemid", "charttime_interval_after_icu_adm", "value" are left
    cur_adm_recordings = cur_adm_recordings[cur_adm_recordings["charttime_interval_after_icu_adm"].isin(range(1, 25))]
    if cur_adm_recordings.shape[0] > 0:
        # only keep the recordings in the first 24-hour window range
        cur_adm_recordings.sort_values(by="charttime_interval_after_icu_adm", inplace=True, ignore_index=True)
        cur_adm_itemids = cur_adm_recordings["itemid"].unique()
        cur_adm_gb_itemid = cur_adm_recordings.groupby("itemid")
        final_cur_adm_recordings = 0
        for j, itemid in enumerate(cur_adm_itemids):
            cur_item = cur_adm_gb_itemid.get_group(itemid)
            cur_item_first_value_one_window = cur_item.groupby("charttime_interval_after_icu_adm").first()
            if j == 0:
                final_cur_adm_recordings = cur_item_first_value_one_window
            else:
                final_cur_adm_recordings = pd.concat([final_cur_adm_recordings, cur_item_first_value_one_window])
        final_cur_adm_recordings.reset_index(inplace=True)
        if dataset_type == "lab":
            final_cur_adm_recordings = final_cur_adm_recordings.pivot(index="charttime_interval_after_icu_adm",
                                                                      columns="itemid", values="valuenum")
        else:  # era == "chart"
            final_cur_adm_recordings = final_cur_adm_recordings.pivot(index="charttime_interval_after_icu_adm",
                                                                      columns="itemid", values="value")

        left_itemids = set(expected_itemids_list) - set(cur_adm_itemids)
        # values in added_itemids_recordings are all NaNs
        added_itemids_recordings = np.empty((final_cur_adm_recordings.shape[0], len(left_itemids)))
        added_itemids_recordings[:] = np.nan
        cur_adm_recordings_added = pd.DataFrame(data=added_itemids_recordings, columns=list(left_itemids),
                                                index=final_cur_adm_recordings.index)
        final_cur_adm_recordings = pd.concat([final_cur_adm_recordings, cur_adm_recordings_added], axis=1)
        final_cur_adm_recordings = final_cur_adm_recordings[expected_itemids_list]
        # add column "time_window"
        final_cur_adm_recordings["time_window"] = final_cur_adm_recordings.index
        first_col = final_cur_adm_recordings.pop("time_window")
        final_cur_adm_recordings.insert(0, "time_window", first_col)
        final_cur_adm_recordings.reset_index(inplace=True, drop=True)
        return final_cur_adm_recordings
    else:  # cur_adm_recordings.shape[0] == 0:
        return cur_adm_recordings



# "recording_range" == 24 (hrs) currently
def Cur_Patient_24hr_Chart_Cat_Itemids_Tabular(cur_adm_recordings, cur_icu_adm_time, era, expected_cols_list, recording_range=24):
    icu_adm_timestamp = datetime.datetime.timestamp(datetime.datetime.strptime(cur_icu_adm_time, "%Y-%m-%d %H:%M:%S"))
    cur_adm_recordings["charttimestamp"] = cur_adm_recordings[["charttime"]].apply((lambda x: datetime.datetime.timestamp(datetime.datetime.strptime(x["charttime"], "%Y-%m-%d %H:%M:%S"))), axis=1)
    cur_adm_recordings[["charttime_interval_after_icu_adm"]] = (cur_adm_recordings[["charttimestamp"]] - icu_adm_timestamp) / 3600
    cur_adm_recordings[["charttime_interval_after_icu_adm"]] = cur_adm_recordings[["charttime_interval_after_icu_adm"]].astype(int) + 1
    # "+1" because astype(int) will only keep the integer part of the float number
    # for example: charttime_interval_after_icu_adm == 0.2, actually this belongs to the first one-hour window,
    # so the result should be 1, not 0 (if returned directly by astype(int))
    cur_adm_recordings.drop(columns=["charttime", "charttimestamp"], inplace=True)
    # columns: "hadm_id", "itemid", "charttime_interval_after_icu_adm", "value" are left
    cur_adm_recordings = cur_adm_recordings[cur_adm_recordings["charttime_interval_after_icu_adm"].isin(range(1, 25))]  # only keep the recordings in the first 24-hour window range
    if cur_adm_recordings.shape[0] > 0:
        cur_adm_recordings.sort_values(by="charttime_interval_after_icu_adm", inplace=True, ignore_index=True)
        cur_adm_itemids = cur_adm_recordings["itemid"].unique()
        cur_adm_gb_itemid = cur_adm_recordings.groupby("itemid")
        final_cur_adm_recordings = 0
        for j, itemid in enumerate(cur_adm_itemids):
            cur_item = cur_adm_gb_itemid.get_group(itemid)
            cur_item_first_value_one_window = cur_item.groupby("charttime_interval_after_icu_adm").first()
            if j == 0:
                final_cur_adm_recordings = cur_item_first_value_one_window
            else:
                final_cur_adm_recordings = pd.concat([final_cur_adm_recordings, cur_item_first_value_one_window])
        final_cur_adm_recordings.reset_index(inplace=True)
        final_cur_adm_recordings = final_cur_adm_recordings.pivot(index="charttime_interval_after_icu_adm", columns="itemid", values="value")

        if era == "CV":
            for cat_itemid in final_cur_adm_recordings.columns:
                cur_itemid_dummies = pd.get_dummies(final_cur_adm_recordings[cat_itemid])
                final_cur_adm_recordings = pd.concat([final_cur_adm_recordings, cur_itemid_dummies], axis=1)
                final_cur_adm_recordings.drop(columns=[cat_itemid], inplace=True, axis=1)
            left_one_hot_cat_itemid_chart_cv = set(expected_cols_list) - set(final_cur_adm_recordings.columns)
            if len(left_one_hot_cat_itemid_chart_cv) > 0:
                added_itemids_recordings = np.zeros((final_cur_adm_recordings.shape[0], len(left_one_hot_cat_itemid_chart_cv))).astype(int)
                final_cur_adm_recording_added = pd.DataFrame(data=added_itemids_recordings,
                                                             columns=list(left_one_hot_cat_itemid_chart_cv),
                                                             index=final_cur_adm_recordings.index)
                final_cur_adm_recordings = pd.concat([final_cur_adm_recordings, final_cur_adm_recording_added], axis=1)
            final_cur_adm_recordings = final_cur_adm_recordings[expected_cols_list]

        else: # era == "MV"
            left_cat_itemid_chart_mv = set(expected_cols_list) - set(final_cur_adm_recordings.columns)
            if len(left_cat_itemid_chart_mv) > 0:
                final_cur_adm_recording_added = pd.DataFrame(np.nan, columns=list(left_cat_itemid_chart_mv), index=final_cur_adm_recordings.index)
                final_cur_adm_recordings = pd.concat([final_cur_adm_recordings, final_cur_adm_recording_added], axis=1)
            final_cur_adm_recordings = final_cur_adm_recordings[expected_cols_list]

        if final_cur_adm_recordings.shape[0] != recording_range:
            cur_index = set(final_cur_adm_recordings.index)
            expected_index = set(range(1, recording_range + 1))
            missed_index = expected_index - cur_index
            if era == "CV":
                # imputation for chart_cv_cat_itemids: only insert those missing 1-hr time windows' rows with all the values: 0
                interpolate_rows = pd.DataFrame(np.zeros((len(missed_index), final_cur_adm_recordings.shape[1])),
                                                columns=final_cur_adm_recordings.columns, index=missed_index)
            else:
                # imputation for chart_cv_cat_itemids: only insert those missing 1-hr time windows' rows with all the values: NaN
                interpolate_rows = pd.DataFrame(np.nan, columns=final_cur_adm_recordings.columns, index=missed_index)
            final_cur_adm_recordings = pd.concat([final_cur_adm_recordings, interpolate_rows]).sort_index(ascending=True)
        # add column "time_window"
        final_cur_adm_recordings["time_window"] = final_cur_adm_recordings.index
        first_col = final_cur_adm_recordings.pop("time_window")
        final_cur_adm_recordings.insert(0, "time_window", first_col)
        final_cur_adm_recordings.reset_index(inplace=True, drop=True)
        return final_cur_adm_recordings
    else:
        return cur_adm_recordings




