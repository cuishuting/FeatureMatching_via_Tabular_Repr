import pandas as pd
from Transform_to_Tabular_utils import Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular
from Imputation import Imputation
pd.options.mode.chained_assignment = None

mimic_icu_adm = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_icuadmittime.csv")
static_mv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_static_MV_added_onetime_itemid.csv")
static_cv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_static_CV.csv")
mimic_lab_mv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_lab_MV_reg_pre_new.csv")
mimic_lab_cv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_lab_CV_reg_pre_new.csv")
mimic_chart_cv_only_num = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_chart_CV_only_numeric_new.csv")
mimic_chart_mv_only_num = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_chart_MV_only_num_del_onetime_itemid.csv")
expected_itemids_list_lab = list(set(mimic_lab_mv["itemid"].unique()).intersection(set(mimic_lab_cv["itemid"].unique())))
expected_itemid_list_chart_cv_only_num = list(mimic_chart_cv_only_num["itemid"].unique())
expected_itemid_list_chart_mv_only_num = list(mimic_chart_mv_only_num["itemid"].unique())
fill_limit_list_lab = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/fill_limit_lab.csv")
fill_limit_chart_mv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/fill_limit_chart_MV.csv")
fill_limit_chart_cv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/fill_limit_chart_CV.csv")
recording_range = 24

"""
MV era
"""
mimic_lab_mv_gb_hadmid = mimic_lab_mv.groupby("hadm_id")
mimic_chart_mv_only_num_gb_hadmid = mimic_chart_mv_only_num.groupby("hadm_id")
whole_hadm_ids_MV = list(static_mv["hadm_id"])
final_tabular_MV_only_num = 0
final_mask_MV_only_num = 0  # same shape as final_tabular_MV_only_num, 0: data missing; 1: have record value
for i, hadm_id in enumerate(whole_hadm_ids_MV):
    if (i+1) % 5000 == 0:
        print("MV: ")
        print(i+1)
    cur_icu_adm_time = mimic_icu_adm[mimic_icu_adm["hadm_id"] == hadm_id]["intime"].unique()[0]

    # lab part: imputed_transformed_tabular_lab_mv
    cur_adm_recordings_lab_mv = mimic_lab_mv_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "valuenum"]]
    transformed_tabular_lab_mv = Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings_lab_mv,
                                                                               cur_icu_adm_time, "lab",
                                                                               expected_itemids_list_lab)

    # chart num part: imputed_transformed_tabular_chart_mv_only_num
    cur_adm_recordings_chart_num_mv = mimic_chart_mv_only_num_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "value"]]
    transformed_tabular_chart_mv_only_num = Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings_chart_num_mv,
                                                                                          cur_icu_adm_time,
                                                                                          "chart",
                                                                                          expected_itemid_list_chart_mv_only_num)
    if transformed_tabular_lab_mv.shape[0] == 0 or transformed_tabular_chart_mv_only_num.shape[0] == 0:  # delete the invalid tabular data
        continue
    else:
        # """Get mask matrix for each tabular for later rnn model use"""
        # expected_tw_list = set(range(1, recording_range+1))
        # """Get cur patient's mask_lab_mv"""
        # cur_adm_mask_lab_mv = pd.DataFrame(0, columns=transformed_tabular_lab_mv.columns, index=transformed_tabular_lab_mv.index)
        # cur_adm_mask_lab_mv.where(transformed_tabular_lab_mv.isnull(), 1, inplace=True)
        # if len(cur_adm_mask_lab_mv["time_window"]) != recording_range:
        #     cur_tw_lab_mv = set(cur_adm_mask_lab_mv["time_window"])
        #     cur_inter_rows_mask_lab_mv = pd.DataFrame(0, index=expected_tw_list-cur_tw_lab_mv, columns=cur_adm_mask_lab_mv.columns[1:])
        #     cur_inter_rows_mask_lab_mv["time_window"] = cur_inter_rows_mask_lab_mv.index
        #     first_col_mask_lab_mv = cur_inter_rows_mask_lab_mv.pop("time_window")
        #     cur_inter_rows_mask_lab_mv.insert(0, "time_window", first_col_mask_lab_mv)
        #     final_mask_lab_mv = pd.concat([cur_adm_mask_lab_mv, cur_inter_rows_mask_lab_mv]).sort_values(by="time_window")
        #     final_mask_lab_mv.reset_index(inplace=True, drop=True)
        # else:
        #     final_mask_lab_mv = cur_adm_mask_lab_mv
        # """Finish Get cur patient's mask_lab_mv"""
        #
        # """Get cur patient's mask_chart_mv_only_num"""
        # cur_adm_mask_chart_mv_only_num = pd.DataFrame(0, columns=transformed_tabular_chart_mv_only_num.columns, index=transformed_tabular_chart_mv_only_num.index)
        # cur_adm_mask_chart_mv_only_num.where(transformed_tabular_chart_mv_only_num.isnull(), 1, inplace=True)
        # if len(cur_adm_mask_chart_mv_only_num["time_window"]) != recording_range:
        #     cur_tw_chart_mv_only_num = set(cur_adm_mask_chart_mv_only_num["time_window"])
        #     cur_inter_rows_mask_chart_mv_only_num = pd.DataFrame(0, index=expected_tw_list-cur_tw_chart_mv_only_num, columns=cur_adm_mask_chart_mv_only_num.columns[1:])
        #     cur_inter_rows_mask_chart_mv_only_num["time_window"] = cur_inter_rows_mask_chart_mv_only_num.index
        #     first_col_mask_chart_mv_only_num = cur_inter_rows_mask_chart_mv_only_num.pop("time_window")
        #     cur_inter_rows_mask_chart_mv_only_num.insert(0, "time_window", first_col_mask_chart_mv_only_num)
        #     final_mask_chart_mv_only_num = pd.concat([cur_adm_mask_chart_mv_only_num, cur_inter_rows_mask_chart_mv_only_num]).sort_values(by="time_window")
        #     final_mask_chart_mv_only_num.reset_index(inplace=True, drop=True)
        # else:
        #     final_mask_chart_mv_only_num = cur_adm_mask_chart_mv_only_num
        # """Finish Get cur patient's mask_chart_mv_only_num"""

        imputed_transformed_tabular_lab_mv = Imputation(fill_limit_list_lab, transformed_tabular_lab_mv, "MV", "lab", 24)
        imputed_transformed_tabular_chart_mv_only_num = Imputation(fill_limit_chart_mv, transformed_tabular_chart_mv_only_num, "MV", "chart", 24)
        # concatenate : imputed_transformed_tabular_lab_mv + imputed_transformed_tabular_chart_mv_only_num
        cur_adm_tabular = pd.concat([imputed_transformed_tabular_lab_mv, imputed_transformed_tabular_chart_mv_only_num.drop("time_window", axis=1)], axis=1)
        if type(final_tabular_MV_only_num) == int:  # final_tabular_MV_only_num == 0
            final_tabular_MV_only_num = cur_adm_tabular
        else:
            final_tabular_MV_only_num = pd.concat([final_tabular_MV_only_num, cur_adm_tabular], axis=0)

final_tabular_MV_only_num.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Train_Data/final_tabular_MV_only_num.csv", index=False)
print("Training data for MV is ready")
"""
CV era
"""
mimic_lab_cv_gb_hadmid = mimic_lab_cv.groupby("hadm_id")
mimic_chart_cv_only_num_gb_hadmid = mimic_chart_cv_only_num.groupby("hadm_id")
whole_hadm_ids_CV = list(static_cv["hadm_id"])
final_tabular_CV_only_num = 0
for i, hadm_id in enumerate(whole_hadm_ids_CV):
    if (i+1) % 5000 == 0:
        print("CV: ")
        print(i+1)
    cur_icu_adm_time = mimic_icu_adm[mimic_icu_adm["hadm_id"] == hadm_id]["intime"].unique()[0]
    # lab part
    cur_adm_recordings_lab_cv = mimic_lab_cv_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "valuenum"]]
    transformed_tabular_lab_cv = Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings_lab_cv,
                                                                               cur_icu_adm_time, "lab",
                                                                               expected_itemids_list_lab)
    # chart num part
    cur_adm_recordings_chart_num_cv = mimic_chart_cv_only_num_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "value"]]
    transformed_tabular_chart_cv_only_num = Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings_chart_num_cv,
                                                                                          cur_icu_adm_time,
                                                                                          "chart",
                                                                                          expected_itemid_list_chart_cv_only_num)
    if transformed_tabular_lab_cv.shape[0] == 0 or transformed_tabular_chart_cv_only_num.shape[0] == 0:
        continue
    else:
        imputed_transformed_tabular_lab_cv = Imputation(fill_limit_list_lab, transformed_tabular_lab_cv, "CV", "lab", 24)
        imputed_transformed_tabular_chart_cv_only_num = Imputation(fill_limit_chart_cv, transformed_tabular_chart_cv_only_num, "CV", "chart", 24)
        # concatenation
        cur_adm_tabular = pd.concat([imputed_transformed_tabular_lab_cv, imputed_transformed_tabular_chart_cv_only_num.drop("time_window", axis=1)], axis=1)
        if type(final_tabular_CV_only_num) == int:  # final_tabular_CV_only_num == 0
            final_tabular_CV_only_num = cur_adm_tabular
        else:
            final_tabular_CV_only_num = pd.concat([final_tabular_CV_only_num, cur_adm_tabular], axis=0)

final_tabular_CV_only_num.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Train_Data/final_tabular_CV_only_num.csv", index=False)
print("Training data for CV is ready")