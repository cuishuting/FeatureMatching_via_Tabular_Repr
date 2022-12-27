import pandas as pd
from Transform_to_Tabular_utils import Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular
from Transform_to_Tabular_utils import Cur_Patient_24hr_Chart_Cat_Itemids_Tabular
from Imputation import Imputation


mimic_icu_adm = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_icuadmittime.csv")
static_mv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_static_MV_added_onetime_itemid.csv")
static_cv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_static_CV.csv")
mimic_lab_mv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_lab_MV_reg_pre_new.csv")
mimic_lab_cv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_lab_CV_reg_pre_new.csv")
mimic_chart_cv_only_num = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_chart_CV_only_numeric_new.csv")
mimic_chart_mv_only_num = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_chart_MV_only_num_del_onetime_itemid.csv")
mimic_chart_mv_only_cat = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_chart_MV_only_cat.csv")
mimic_chart_cv_only_cat = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_chart_CV_only_cat.csv")
expected_itemids_list_lab = list(set(mimic_lab_mv["itemid"].unique()).intersection(set(mimic_lab_cv["itemid"].unique())))
expected_itemid_list_chart_cv_only_num = list(mimic_chart_cv_only_num["itemid"].unique())
expected_itemid_list_chart_mv_only_num = list(mimic_chart_mv_only_num["itemid"].unique())
expected_cols_list_chart_mv_only_cat = list(mimic_chart_mv_only_cat["itemid"].unique())
chart_cv_only_cat_itemids_value_map_list = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/chart_cv_cat_itemids_unique_value_map.csv")
expected_cols_list_chart_cv_only_cat = list(chart_cv_only_cat_itemids_value_map_list["mapped_value"].unique())
fill_limit_list_lab = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/fill_limit_lab.csv")
fill_limit_chart_mv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/fill_limit_chart_MV.csv")
fill_limit_chart_cv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/fill_limit_chart_CV.csv")
"""
First Generate the Test data set using the first 100 patients (hadm_id) 
"""
"""
MV era
"""
mimic_lab_mv_gb_hadmid = mimic_lab_mv.groupby("hadm_id")
mimic_chart_mv_only_num_gb_hadmid = mimic_chart_mv_only_num.groupby("hadm_id")
mimic_chart_mv_only_cat_gb_hadmid = mimic_chart_mv_only_cat.groupby("hadm_id")
test_100_hadm_ids_MV = list(static_mv["hadm_id"][:100])
final_test_tabular_MV = 0
for i, hadm_id in enumerate(test_100_hadm_ids_MV):
    # static part: cur_static
    cur_static = pd.concat([static_mv[static_mv["hadm_id"] == hadm_id]]*24, ignore_index=True)
    cur_static.drop("hadm_id", axis=1, inplace=True)

    cur_icu_adm_time = mimic_icu_adm[mimic_icu_adm["hadm_id"] == hadm_id]["intime"].unique()[0]
    # lab part: imputed_transformed_tabular_lab_mv
    cur_adm_recordings_lab_mv = mimic_lab_mv_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "valuenum"]]
    transformed_tabular_lab_mv = Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings_lab_mv, cur_icu_adm_time, "lab", expected_itemids_list_lab)
    imputed_transformed_tabular_lab_mv = Imputation(fill_limit_list_lab, transformed_tabular_lab_mv, "MV", "lab", 24)

    # chart num part: imputed_transformed_tabular_chart_mv_only_num
    cur_adm_recordings_chart_num_mv = mimic_chart_mv_only_num_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "value"]]
    transformed_tabular_chart_mv_only_num = Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings_chart_num_mv, cur_icu_adm_time, "chart", expected_itemid_list_chart_mv_only_num)
    imputed_transformed_tabular_chart_mv_only_num = Imputation(fill_limit_chart_mv, transformed_tabular_chart_mv_only_num, "MV", "chart", 24)

    # chart cat part: transformed_tabular_chart_mv_only_cat
    cur_adm_recordings_chart_cat_mv = mimic_chart_mv_only_cat_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "value"]]
    transformed_tabular_chart_mv_only_cat = Cur_Patient_24hr_Chart_Cat_Itemids_Tabular(cur_adm_recordings_chart_cat_mv, cur_icu_adm_time, "MV", expected_cols_list_chart_mv_only_cat, 24)

    # concatenate : imputed_transformed_tabular_lab_mv + imputed_transformed_tabular_chart_mv_only_num + transformed_tabular_chart_mv_only_cat + cur_static
    cur_adm_tabular = pd.concat([imputed_transformed_tabular_lab_mv, imputed_transformed_tabular_chart_mv_only_num.drop("time_window", axis=1), transformed_tabular_chart_mv_only_cat.drop("time_window", axis=1), cur_static], axis=1)
    if i == 0:
        final_test_tabular_MV = cur_adm_tabular
    else:
        final_test_tabular_MV = pd.concat([final_test_tabular_MV, cur_adm_tabular], axis=0)

final_test_tabular_MV.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Test_Data/test_tabular_repre_MV_100.csv", index=False)

"""
CV era
"""
mimic_lab_cv_gb_hadmid = mimic_lab_cv.groupby("hadm_id")
mimic_chart_cv_only_num_gb_hadmid = mimic_chart_cv_only_num.groupby("hadm_id")
mimic_chart_cv_only_cat_gb_hadm_id = mimic_chart_cv_only_cat.groupby("hadm_id")
test_100_hadm_ids_CV = list(static_cv["hadm_id"][:100])
final_test_tabular_CV = 0
for i, hadm_id in enumerate(test_100_hadm_ids_CV):
    # static part
    cur_static = pd.concat([static_cv[static_cv["hadm_id"] == hadm_id]]*24, ignore_index=True)
    cur_static.drop("hadm_id", axis=1, inplace=True)

    cur_icu_adm_time = mimic_icu_adm[mimic_icu_adm["hadm_id"] == hadm_id]["intime"].unique()[0]
    # lab part
    cur_adm_recordings_lab_cv = mimic_lab_cv_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "valuenum"]]
    transformed_tabular_lab_cv = Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings_lab_cv, cur_icu_adm_time, "lab", expected_itemids_list_lab)
    imputed_transformed_tabular_lab_cv = Imputation(fill_limit_list_lab, transformed_tabular_lab_cv, "CV", "lab", 24)

    # chart num part
    cur_adm_recordings_chart_num_cv = mimic_chart_cv_only_num_gb_hadmid.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "value"]]
    transformed_tabular_chart_cv_only_num = Cur_Patient_24hr_Lab_ChartNum_Itemids_Tabular(cur_adm_recordings_chart_num_cv, cur_icu_adm_time, "chart", expected_itemid_list_chart_cv_only_num)
    imputed_transformed_tabular_chart_cv_only_num = Imputation(fill_limit_chart_cv, transformed_tabular_chart_cv_only_num, "CV", "chart", 24)

    # chart cat part
    cur_adm_recordings_chart_cat_cv = mimic_chart_cv_only_cat_gb_hadm_id.get_group(hadm_id)[["hadm_id", "itemid", "charttime", "value"]]
    transformed_tabular_chart_cv_only_cat = Cur_Patient_24hr_Chart_Cat_Itemids_Tabular(cur_adm_recordings_chart_cat_cv, cur_icu_adm_time, "CV", expected_cols_list_chart_cv_only_cat, 24)

    # concatenate:
    cur_adm_tabular = pd.concat([imputed_transformed_tabular_lab_cv, imputed_transformed_tabular_chart_cv_only_num.drop("time_window", axis=1), transformed_tabular_chart_cv_only_cat.drop("time_window", axis=1), cur_static], axis=1)
    if i == 0:
        final_test_tabular_CV = cur_adm_tabular
    else:
        final_test_tabular_CV = pd.concat([final_test_tabular_CV, cur_adm_tabular], axis=0)
final_test_tabular_CV.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Test_Data/test_tabular_repre_CV_100.csv", index=False)









