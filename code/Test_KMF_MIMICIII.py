import pandas as pd
import numpy as np
from sklearn import model_selection
from KMF_only_module import Simple_maximum_sim_viaCorrelation
np.seterr(divide='ignore')

data_mv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Train_Data/final_tabular_MV_only_num.csv")
data_cv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Train_Data/final_tabular_CV_only_num.csv")
num_of_patients_mv = data_mv.shape[0] / 24
num_of_patients_cv = data_cv.shape[0] / 24
mimic_lab_mv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_lab_MV_reg_pre_new.csv")
mimic_lab_cv = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_lab_CV_reg_pre_new.csv")
data_mv_no_tw = data_mv.drop(columns="time_window")
data_cv_no_tw = data_cv.drop(columns="time_window")

# "mapped_features" is a list with 36 elements, each is the mapped itemid with dtype: int
mapped_features = list(set(mimic_lab_mv["itemid"].unique()).intersection(set(mimic_lab_cv["itemid"].unique())))
mapped_features = [str(i) for i in mapped_features]

""" some initial things related to MIMIC """

# Getting list of all items along with the source and label
item_id_dbsource = pd.read_csv('/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Org_Data/d_items_chartevents.csv')
itemid_labs = pd.read_csv('/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Org_Data/d_items_labevents.csv')

# creating label dictionary for later
itemid_label_lab = dict(zip(list(itemid_labs.itemid), list(itemid_labs.label)))
itemid_label_chart = dict(zip(list(item_id_dbsource.itemid), list(item_id_dbsource.label)))
itemid_label_dict = {**itemid_label_chart, **itemid_label_lab}  # merging two dictionaries here

""" preparing true match matrix """

# Itemids with match in order of matching (ultiple case as of now; will be removed in a while)
CV_itemids_with_match_t = [211, 8549, 5815, 51, 8368, 52, 5813, 8547, 113, 455, 8441, 456, 618, 779, 834, 814, 778, 646,
                           506, 813, 861, 1127, 1542, 770, 788, 1523, 791, 1525, 811, 821, 1532, 769, 1536, 8551, 5817,
                           678, 8554, 5820, 780, 1126, 470, 190, 50, 8553, 5819, 763, 683, 682, 684, 450, 619, 614, 615,
                           535, 543, 444, 578, 776, 773, 1162, 781, 786, 1522, 784, 796, 797, 798, 799, 800, 807, 816,
                           818, 1531, 827, 1534, 848, 1538, 777, 762, 837, 1529, 920, 1535, 785, 772, 828, 829, 1286,
                           824, 1533, 825, 1530, 815, 6206, 6207]  # length: 95
MV_itemids_with_match_t = [220045, 220046, 220047, 220050, 220051, 220052, 220056, 220058, 220074, 220179, 220180,
                           220181, 220210, 220224, 220227, 220228, 220235, 220277, 220339, 220545, 220546, 220546,
                           220546, 220587, 220602, 220602, 220615, 220615, 220621, 220635, 220635, 220644, 220645,
                           223751, 223752, 223761, 223769, 223770, 223830, 223830, 223834, 223835, 223876, 224161,
                           224162, 224639, 224684, 224685, 224686, 224687, 224688, 224689, 224690, 224695, 224696,
                           224697, 224701, 224828, 225612, 225624, 225624, 225625, 225625, 225634, 225639, 225640,
                           225641, 225642, 225643, 225664, 225667, 225668, 225668, 225677, 225677, 225690, 225690,
                           225698, 226512, 226534, 226537, 226707, 227442, 227445, 227456, 227457, 227464, 227465,
                           227465, 227466, 227466, 227467, 227467, 227565, 227566]  # length: 95


# converting the above integers into strings
CV_itemids_with_match_t = [str(i) for i in CV_itemids_with_match_t]
MV_itemids_with_match_t = [str(i) for i in MV_itemids_with_match_t]

match_df = pd.DataFrame(columns=['CV_itemids', 'CV_labels', 'MV_itemids', 'MV_labels'])
match_df['CV_itemids'] = CV_itemids_with_match_t
match_df['MV_itemids'] = MV_itemids_with_match_t
for i in range(len(match_df)):
    match_df.loc[i, "CV_labels"] = itemid_label_dict[int(match_df.loc[i, 'CV_itemids'])]
    match_df.loc[i, "MV_labels"] = itemid_label_dict[int(match_df.loc[i, 'MV_itemids'])]

# removing the rows that are beyond one to one matching
match_df.drop_duplicates(subset=['MV_itemids'], inplace=True)

CV_itemids_with_match = list(match_df['CV_itemids'])
MV_itemids_with_match = list(match_df['MV_itemids'])

CV_itemids_to_drop = [i for i in CV_itemids_with_match_t if i not in CV_itemids_with_match]
# "CV_itemids_to_drop" are those itemids represent those duplicate MV's itemids' match
chart_cv_only_num = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_chart_CV_only_numeric_new.csv")
chart_mv_only_num = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Processed_Data/mimic_chart_MV_only_num_del_onetime_itemid.csv")
onlychart_cont_CV = list(chart_cv_only_num["itemid"].unique())
onlychart_cont_MV = list(chart_mv_only_num["itemid"].unique())
onlychart_cont_CV = [str(i) for i in onlychart_cont_CV]
onlychart_cont_MV = [str(i) for i in onlychart_cont_MV]
""" something to fill inbetween to take care of the duplicate labs in chartevents """
# run the follwing routine once only since it is inplace
for i in CV_itemids_to_drop:
    # onlychart_CV.remove(str(i))
    if i in onlychart_cont_CV:
        onlychart_cont_CV.remove(str(i))

labeled_lab_itemid = pd.read_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/CheckPoint_Files/GS_match_result_Ryan_lab_noted.csv")
fake_chart_itemid_MV = labeled_lab_itemid[labeled_lab_itemid["islab_MV"]==1]["ump_itemid_in_MV"]
fake_chart_itemid_CV = labeled_lab_itemid[labeled_lab_itemid["islab_CV"]==1]["match_byGS_in_CV"]
fake_chart_itemid_MV = [str(i) for i in fake_chart_itemid_MV]
fake_chart_itemid_CV = [str(i) for i in fake_chart_itemid_CV]

match_df.drop(match_df[match_df['CV_itemids'].isin(fake_chart_itemid_CV)].index, inplace=True)
match_df.drop(match_df[match_df['MV_itemids'].isin(fake_chart_itemid_MV)].index, inplace=True)

CV_itemids_with_match = list(match_df['CV_itemids'])
MV_itemids_with_match = list(match_df['MV_itemids'])

for i in fake_chart_itemid_CV:
    # onlychart_CV.remove(str(i))
    if i in onlychart_cont_CV:
        onlychart_cont_CV.remove(str(i))

for i in fake_chart_itemid_MV:
    # onlychart_MV.remove(str(i))
    if i in onlychart_cont_MV:
        onlychart_cont_MV.remove(str(i))

print("before drop: ")
print(data_cv_no_tw.shape)  # (624768, 138)
print(data_mv_no_tw.shape)  # (503232, 119)
data_cv_no_tw.drop(columns=fake_chart_itemid_CV, inplace=True)
data_mv_no_tw.drop(columns=fake_chart_itemid_MV, inplace=True)
data_mv_no_tw.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Train_Data/final_tab_MV_no_tw_no_fake_chart.csv", index=False)
data_cv_no_tw.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/Train_Data/final_tab_CV_no_tw_no_fake_chart.csv", index=False)

print("after drop: ")
print(data_cv_no_tw.shape)  # (624768, 101)
print(data_mv_no_tw.shape)  # (503232, 82)
match_dic = dict(zip(MV_itemids_with_match, CV_itemids_with_match))


# itemids with no match
CV_itemids_withnomatch = [i for i in onlychart_cont_CV if i not in CV_itemids_with_match]
# onlychart_cont_CV is the set of chartevents from CV
MV_itemids_withnomatch = [i for i in onlychart_cont_MV if i not in MV_itemids_with_match]
# onlychart_cont_MV is the set of chartevents from MV

print(" CV_itemids_with match ", len(CV_itemids_with_match))  # 45
print(" MV_itemids_with match ", len(MV_itemids_with_match))  # 45

print(" CV_itemids_with NO match ", len(CV_itemids_withnomatch))  # 25
print(" MV_itemids_with NO match ", len(MV_itemids_withnomatch))  # 10

num_xtra_feat_inX1 = len(MV_itemids_withnomatch)

train_size_mv = int(num_of_patients_mv * 0.8) * 24
train_size_cv = int(num_of_patients_cv * 0.8) * 24
data_mv_tr, data_mv_holdout = model_selection.train_test_split(data_mv_no_tw, train_size=train_size_mv, shuffle=False)
data_cv_tr, data_cv_holdout = model_selection.train_test_split(data_cv_no_tw, train_size=train_size_cv, shuffle=False)

""" # true permutation matrix  """


P_x1 = np.zeros((len(data_mv_tr.columns),
                 len(data_cv_tr.columns)))
print("Shape of P_x1 ", P_x1.shape)

for i in range(len(data_mv_tr.columns)):
    for j in range(len(data_cv_tr.columns)):
        if data_mv_tr.columns[i] == data_cv_tr.columns[j]:
            P_x1[i, j] = 1
        elif data_mv_tr.columns[i] in MV_itemids_with_match:
            if (match_dic[data_mv_tr.columns[i]] == data_cv_tr.columns[j]):
                P_x1[i, j] = 1
        elif (data_mv_tr.columns[i] in MV_itemids_withnomatch) & (data_cv_tr.columns[j] in CV_itemids_withnomatch):
            P_x1[i, j] = 2

np.save("/storage1/christopherking/Active/mimic3/c.shuting/MIMICIII_DeepRecurrent_Models/Permutation_Matrix_both_mp_ump.npy", P_x1)
cor_btw_df = np.zeros((len(data_mv_tr.columns), len(data_cv_tr.columns)))
for i, col in enumerate(data_cv_tr.columns):
    cor_btw_df[:, i] = data_mv_tr.corrwith(data_cv_tr[col])
cor_df = pd.DataFrame(cor_btw_df, index=data_mv_tr.columns, columns=data_cv_tr.columns)
cor_df.fillna(0, inplace=True)
_, _, _, _, temp_inf_x1, temp_inf_x2, _, _ = Simple_maximum_sim_viaCorrelation(data_mv_tr.copy(), data_cv_tr.copy(), P_x1[len(mapped_features):, len(mapped_features):], data_mv_tr.columns, data_cv_tr.columns,
                                  mapped_features, cor_df, data_mv_holdout, data_cv_holdout, num_xtra_feat_inX1)

selected_itemid_dbsource_for_ump_x1 = item_id_dbsource[["itemid", "label"]].astype({"itemid": "int64"})
selected_itemid_dbsource_for_ump_x1.rename(columns={"itemid": "ump_feature_in_X1", "label": "ump_feature_in_X1_label"}, inplace=True)
temp_inf_x1 = temp_inf_x1.astype({"ump_feature_in_X1": "int64"})
new_inf_x1 = temp_inf_x1.merge(selected_itemid_dbsource_for_ump_x1, how="inner", on="ump_feature_in_X1")

selected_itemid_dbsource_for_predicted_matched_byGS = item_id_dbsource[["itemid", "label"]].astype({"itemid": "int64"})
selected_itemid_dbsource_for_predicted_matched_byGS.rename(columns={"itemid": "match_byGS", "label": "match_byGS_label"}, inplace=True)
temp_inf_x2 = temp_inf_x2.astype({"match_byGS": "int64"})
new_inf_x2 = temp_inf_x2.merge(selected_itemid_dbsource_for_predicted_matched_byGS, how="inner", on="match_byGS")

matched_pairs_itemid_label = pd.concat([new_inf_x1[["ump_feature_in_X1", "ump_feature_in_X1_label", "match_byGS"]], new_inf_x2[["match_byGS_label"]]], axis=1)
# print(new_inf_x1)
# print(new_inf_x2)
matched_pairs_itemid_label.to_csv("/storage1/christopherking/Active/mimic3/c.shuting/MIMIC_III_Schema_Matching_Tabular_Repr/CheckPoint_Files/GS_match_result.csv", index=False)








