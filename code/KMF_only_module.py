import numpy as np
from sklearn.metrics import pairwise
from scipy import stats
import pandas as pd
from matching.games import HospitalResident
import sys
import pingouin as pg

print(sys.getrecursionlimit())
sys.setrecursionlimit(3500)
print(sys.getrecursionlimit())


def Covaraince_matrix_factor_method(dimension, factors):
    n = dimension
    k = factors
    W = np.random.standard_normal(size=(n, k))  # k< n and hence not full rank
    print("W matrix")
    # print(W)
    list_random = np.random.randint(1, 20, n)
    # list_random = np.random.randint(100,200, n)  # only for dataset 8
    print("List of random numbers used to generate the diagonal matrix for covariance")
    print(list_random)
    # exit()
    D = np.identity(n)
    D[np.diag_indices_from(D)] = list_random
    cov = np.matmul(W, np.transpose(W)) + D  # adding D to make the matrix full rank

    return cov


def Matching_via_HRM(C_X1_train, C_X2_train, P_x1_O_to_R, num_mapped_axis):  # in this case here the small feature sized database is X1, so we need to treat it as hospital and there will be capacities on it.
    ####### ----------  X1 train ------------- ##########

    true_features_pref_X1_train = {}
    cross_recon_features_pref_X1_train = {}
    capacities_X1_train = {}

    for i in range(C_X1_train.shape[0]):  # C_X1_train.shape[0]: number of unmapped features in dataset_1
        sorted_index = np.argsort(-C_X1_train[i, :])
        # "sorted_index" is ranked based on the most similar to the least similar order (the value is the cosine similarity,
        # the closer the 2 features(angle btw 2 features), the closer the cosine similarity is to 1
        # the farther the 2 features(angle btw 2 features), the closer the cosine similarity is to -1)
        # between current unmapped feature "i" in dataset_1 and each unmapped feature in dataset_2(the cols of C_X1_train)
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        # "sorted_col_index" is the rank list of unmapped features in dataset_2(true_col_index+1) based on
        # similarity between current unmapped feature i in dataset_1 and each unmapped feature in dataset_2
        true_features_pref_X1_train["R" + str(i + 1)] = sorted_col_index
        capacities_X1_train["R" + str(i + 1)] = 1

    for j in range(C_X1_train.shape[1]): # C_X1_train.shape[1]:  number of unmapped features in dataset_2
        sorted_index = np.argsort(-C_X1_train[:, j])
        # "sorted_index" is ranked based on the most similar to the least similar order
        # between current unmapped feature "j" in dataset_2 and each unmapped feature in dataset_1 (the rows of C_X1_train)
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        # "sorted_col_index" is the rank list of unmapped features in dataset_1(true_row_index+1) based on
        # similarity between current unmapped feature j in dataset_2 and each unmapped features in dataset_1
        cross_recon_features_pref_X1_train["C" + str(j + 1)] = sorted_col_index

    game_X1_train = HospitalResident.create_from_dictionaries(cross_recon_features_pref_X1_train,
                                                              true_features_pref_X1_train,
                                                              capacities_X1_train)

    ####### ----------  X2 train ------------- ##########

    true_features_pref_X2_train = {}
    cross_recon_features_pref_X2_train = {}
    capacities_X2_train = {}

    for i in range(C_X2_train.shape[0]):  # C_X2_train.shape[0]: number of unmapped features in dataset_2
        sorted_index = np.argsort(-C_X2_train[i, :])
        # sorted_index: the rank list from most similar to least similar between current unmapped feature i in dataset_2
        # and each unmapped feature in dataset_1
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X2_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X2_train.shape[1]):  # C_X2_train.shape[1]: number of unmapped features in dataset_1
        sorted_index = np.argsort(-C_X2_train[:, j])
        # sorted_index: the rank list from most similar to least similar between current unmapped feature j in dataset_1
        # and each unmapped feature in dataset_2
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X2_train["C" + str(j + 1)] = sorted_col_index
        capacities_X2_train["C" + str(j + 1)] = 1

    # create_from_dictionaries(resident_prefs, hospital_prefs, capacities, clean=False)
    game_X2_train = HospitalResident.create_from_dictionaries(true_features_pref_X2_train,
                                                              cross_recon_features_pref_X2_train,
                                                              capacities_X2_train)

       ######   ------------  Final matching -----------   ##########

    print("\n ------- Matching from X1_train  --------- \n")
    matching_x1_train = game_X1_train.solve()
    print(matching_x1_train)

    print("\n ------- Matching from X2_train  --------- \n")
    matching_x2_train = game_X2_train.solve()
    print(matching_x2_train)
    x1_train_y = [int(str(v[0])[1:]) if v else None for v in matching_x1_train.values()]
    x2_train_y = [int(str(v[0])[1:]) if v else None for v in matching_x2_train.values()]

    # matching matrices
    matching_x1_train_matrix = np.zeros(C_X1_train.shape)
    # shape: [num_unmapped_features_in_d1, num_unmapped_features_in_d2]
    matching_x2_train_matrix = np.zeros(np.transpose(C_X2_train).shape)
    # shape: [num_unmapped_features_in_d1, num_unmapped_features_in_d2]

    for i in range(matching_x1_train_matrix.shape[0]):  # number of unmapped features in d_1
        if x1_train_y[i] is not None:
            matching_x1_train_matrix[i, x1_train_y[i] - 1] = 1  # shape: [# of ump features in d1, # of ump features in d2]
        # unmapped feature i in d_1 and unmapped feature "x1_train_y[i] - 1" in d_2 has a match

    for i in range(matching_x2_train_matrix.shape[0]):  # number of unmapped features in d_1
        if x2_train_y[i] is not None:
            matching_x2_train_matrix[i, x2_train_y[i] - 1] = 1  # shape: [# of ump features in d1, # of ump features in d2]
        # unmapped feature i in d_1 and unmapped feature "x2_train_y[i] - 1" in d_2 has a match
    # getting the number of correct matches that had a match in other database
    num_correct_from_x1 = 0
    num_correct_from_x2 = 0
    for i in range(P_x1_O_to_R.shape[0]):  # number of unmapped features in d_1
        if np.all(P_x1_O_to_R[i] == matching_x1_train_matrix[i]):
            # only when the positions of 0-1 are exactly the same, will this condition be true
            num_correct_from_x1 = num_correct_from_x1 + 1
        if np.all(P_x1_O_to_R[i] == matching_x2_train_matrix[i]):
            num_correct_from_x2 = num_correct_from_x2 + 1

    return num_correct_from_x1, num_correct_from_x2, matching_x1_train_matrix, matching_x2_train_matrix


def Simple_maximum_sim_viaCorrelation(df_train_preproc, df_rename_preproc, P_x1, reordered_column_names_orig, reordered_column_names_r, mapped_features, Cor_from_df, Df_holdout_orig, DF_holdout_r, num_xtra_feat_inX1):
    """
    :param df_train_preproc:  dataset 1 training set : data_mv
    :param df_rename_preproc: dataset 2 trainng set : data_cv
    :param P_x1:  true permutation matrix between only unmatched features
    :param reordered_column_names_orig:  column names of dataset 1
    :param reordered_column_names_r: column names of dataset 2
    :param mapped_features: names of mapped features common between dataset 1 nd dataset 2
    :param Cor_from_df:correlation between the features of two datasets (includes both mapped and unmapped)
    :param Df_holdout_orig: dataset 1 holdout set for bootstrapping
    :param DF_holdout_r: dataset 2 holdout set for bootstrapping
    :return:
    """

    mpfeatures = len(mapped_features)
    unmapped_features_orig = [i for i in reordered_column_names_orig if i not in mapped_features]  # 83
    unmapped_features_r = [i for i in reordered_column_names_r if i not in mapped_features]  # 102
    CorMatrix_X1_unmap_mapped = df_train_preproc.corr().loc[unmapped_features_orig, mapped_features]  # [83, 36]
    CorMatrix_X2_unmap_mapped = df_rename_preproc.corr().loc[unmapped_features_r, mapped_features]  # [102, 36]
    print("Checkpoint 1")

    # similarity between the correlation matrices

    if np.any(np.isnan(CorMatrix_X1_unmap_mapped)) == True or np.any(np.isnan(CorMatrix_X2_unmap_mapped)) == True:
        CorMatrix_X1_unmap_mapped = np.nan_to_num(CorMatrix_X1_unmap_mapped)
        CorMatrix_X2_unmap_mapped = np.nan_to_num(CorMatrix_X2_unmap_mapped)
        print("Here here")

    sim_cor_norm_X1_to_X2 = pairwise.cosine_similarity(CorMatrix_X1_unmap_mapped, CorMatrix_X2_unmap_mapped, dense_output=True)
    # shape: [83, 102]
    sim_cor_norm_X2_to_X1 = pairwise.cosine_similarity(CorMatrix_X2_unmap_mapped, CorMatrix_X1_unmap_mapped, dense_output=True)
    # shape: [102, 83]

    print("Checkpoint 2")

    """ Calling the stable marriage algorithm for mappings  """

    correct_with_match_from_x1_test, correct_with_match_from_x2_test, x1_match_matrix_test, x2_match_matrix_test = Matching_via_HRM(sim_cor_norm_X1_to_X2, sim_cor_norm_X2_to_X1, P_x1, len(mapped_features))

    test_statistic_num_fromX1 = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test.shape[0]) for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    # length: 83--number of unmapped features in d_1(d_mv), because the Gale-shapely algorithm gives each unmapped feature in d_1 with a predicted match
    # though the golden-matched-standard only provides 81 matched pairs btw d_cv and d_mv
    test_statistic_num_fromX2 = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test.shape[0]) for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
    # length: 83



    print("Checkpoint 3")

    # Bootstrap samples to obtain the standard deviation methods to be later used in p value computation
    num_of_bts = 10
    # bts_for_allthe_accepted_matches_fromX1/2 stores the cosine similarity btw [each unmapped feature in d_1 and corresponding predicted matched feature in d_2]
    # based on sampled data sampling num_of_bts times from holdout dataset.
    bts_for_allthe_accepted_matches_fromX1 = np.zeros((len(unmapped_features_orig), num_of_bts))
    bts_for_allthe_accepted_matches_fromX2 = np.zeros((len(unmapped_features_orig), num_of_bts))

    for bts in range(num_of_bts):
        Df_holdout_orig_bts = Df_holdout_orig.sample(n=len(Df_holdout_orig), replace=True, random_state=bts, axis=0)
        DF_holdout_r_bts = DF_holdout_r.sample(n=len(DF_holdout_r), replace=True, random_state=bts, axis=0)
        CorMatrix_X1_unmap_mapped_bts = Df_holdout_orig_bts.corr().loc[unmapped_features_orig, mapped_features]
        CorMatrix_X2_unmap_mapped_bts = DF_holdout_r_bts.corr().loc[unmapped_features_r, mapped_features]

        # similarity between the correlation matrices
        if np.any(np.isnan(CorMatrix_X1_unmap_mapped_bts)) == True or np.any(np.isnan(CorMatrix_X2_unmap_mapped_bts)) == True:
            CorMatrix_X1_unmap_mapped_bts = np.nan_to_num(CorMatrix_X1_unmap_mapped_bts)
            CorMatrix_X2_unmap_mapped_bts = np.nan_to_num(CorMatrix_X2_unmap_mapped_bts)
            print("Here here")

        sim_cor_norm_X1_to_X2_bts = pairwise.cosine_similarity(CorMatrix_X1_unmap_mapped_bts, CorMatrix_X2_unmap_mapped_bts, dense_output=True)
        # shape: [num_of_um_features_in_d1, num_of_um_features_in_d2]
        sim_cor_norm_X2_to_X1_bts = pairwise.cosine_similarity(CorMatrix_X2_unmap_mapped_bts, CorMatrix_X1_unmap_mapped_bts, dense_output=True)
        # shape: [num_of_um_features_in_d2, num_of_um_features_in_d1]
        """ Calling the stable marriage algorithm for mappings  """

        # _ ,_ , x1_match_matrix_test_bts, x2_match_matrix_test_bts = Matching_via_HRM(sim_cor_norm_X1_to_X2_bts, sim_cor_norm_X2_to_X1_bts, P_x1, len(mapped_features))

        # we will use the matched found on the whole dataset and use the bootstraps only to get the dot product estimates

        bts_for_allthe_accepted_matches_fromX1[:, bts] = [sim_cor_norm_X1_to_X2_bts[i, j]
                                                          for i in range(x1_match_matrix_test.shape[0])
                                                          for j in range(x1_match_matrix_test.shape[1])
                                                          if x1_match_matrix_test[i, j] == 1]
        # bts_for_allthe_accepted_matches_fromX1/2 shape: [num_of_um_features_in_d1, num_of_bts]
        # the selected cosine similarity are the predicted matched feature pairs from HRM algorithm using the original dataset(dataset1_tr, dataset2_tr) to train
        bts_for_allthe_accepted_matches_fromX2[:, bts] = [sim_cor_norm_X2_to_X1_bts[j, i]
                                                          for i in range(x2_match_matrix_test.shape[0])
                                                          for j in range(x2_match_matrix_test.shape[1])
                                                          if x2_match_matrix_test[i, j] == 1]

    # test_statistic_den_fromX1 are the list with length==num_um_feature_in_d1,
    # each element are the [std of cosine similarities] of each predicted matched feature pair for dataset_1's unmapped features based on boostrap samples
    # test_statistic_den_fromX1/2's length: number of unmapped features in dataset 1
    test_statistic_den_fromX1 = [np.std(bts_for_allthe_accepted_matches_fromX1[i, :]) for i in range(x1_match_matrix_test.shape[0])]
    test_statistic_den_fromX2 = [np.std(bts_for_allthe_accepted_matches_fromX2[i, :]) for i in range(x1_match_matrix_test.shape[0])]

    temp_inf_x1 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])
    temp_inf_x2 = pd.DataFrame(
        columns=['ump_feature_in_X1', "CV_label", 'match_byGS', "match_byGS(MV_label)", 'true_correlation',
                 'estimated_cross_corr', 'corr_p_value', 'SD_rejects_H0', 'no_match_or_not'])

    # getting the p values that needs to be tested for significance
    test_statistic_for_cor_sig_fromX1 = np.array(test_statistic_num_fromX1) / np.array(test_statistic_den_fromX1)
    test_statistic_for_cor_sig_fromX2 = np.array(test_statistic_num_fromX2) / np.array(test_statistic_den_fromX2)

    temp_inf_x1.corr_p_value = [stats.norm.sf(abs(x)) * 2 for x in test_statistic_for_cor_sig_fromX1]
    temp_inf_x2.corr_p_value = [stats.norm.sf(abs(x)) * 2 for x in test_statistic_for_cor_sig_fromX2]

    # 　"estimated_cross_corr": the cosine similarity between predicted matched feature pairs
    temp_inf_x1.estimated_cross_corr = [sim_cor_norm_X1_to_X2[i, j] for i in range(x1_match_matrix_test.shape[0])
                                        for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
    temp_inf_x2.estimated_cross_corr = [sim_cor_norm_X2_to_X1[j, i] for i in range(x2_match_matrix_test.shape[0])
                                        for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]  # (j,i) because the match is from small to large and original p values are from large to small

    # testing whether some of the proposed matches are such that there exist no match in reality but GS assigned one;
    # False in the reject list below(temp_inf_x1.SD_rejects_H0) can be interpreted as the case where the testing procedure says there wasn't any match originally
    # SD_rejects_H0's length: unmapped features in d_1
    temp_inf_x1.SD_rejects_H0, _ = pg.multicomp(np.array(temp_inf_x1.corr_p_value), method='fdr_by', alpha=0.05)
    temp_inf_x2.SD_rejects_H0, _ = pg.multicomp(np.array(temp_inf_x2.corr_p_value), method='fdr_by', alpha=0.05)

    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        matched_index = [j for j in range(x1_match_matrix_test.shape[1]) if x1_match_matrix_test[i, j] == 1]
        # "matched_index" are the feature index in D_2 predicted as the matched feature for feature i in D_1 (so "matched_index" is a list with length 1)
        temp_inf_x1.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        # temp_inf_x1.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x1.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        # temp_inf_x1.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
        #     int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]

        # true correlation of current unmapped feature in d_1 and its predicted matched pair in d_2
        temp_inf_x1.loc[i, "true_correlation"] = Cor_from_df.loc[reordered_column_names_orig[len(mapped_features) + i],
                                                                 reordered_column_names_r[len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x1.loc[i, "no_match_or_not"] = 1
        else:
            temp_inf_x1.loc[i, "no_match_or_not"] = 0

    for i in range(len(temp_inf_x2.SD_rejects_H0)):
        matched_index = [j for j in range(x2_match_matrix_test.shape[1]) if x2_match_matrix_test[i, j] == 1]
        temp_inf_x2.loc[i, "ump_feature_in_X1"] = reordered_column_names_orig[len(mapped_features) + i]
        # temp_inf_x2.loc[i, "CV_label"] = itemid_label_dict[int(reordered_column_names_orig[len(mapped_features) + i])]
        temp_inf_x2.loc[i, "match_byGS"] = reordered_column_names_r[len(mapped_features) + matched_index[0]]
        # temp_inf_x2.loc[i, "match_byGS(MV_label)"] = itemid_label_dict[
        #     int(reordered_column_names_r[len(mapped_features) + matched_index[0]])]
        temp_inf_x2.loc[i, "true_correlation"] = Cor_from_df.loc[reordered_column_names_orig[len(mapped_features) + i],
                                                                 reordered_column_names_r[len(mapped_features) + matched_index[0]]]
        if np.any(P_x1[i] == 2):
            temp_inf_x2.loc[i, "no_match_or_not"] = 1
        else:
            temp_inf_x2.loc[i, "no_match_or_not"] = 0

    correct_with_no_match_from_CCx1_test = 0
    correct_with_no_match_from_CCx2_test = 0
    for i in range(len(temp_inf_x1.SD_rejects_H0)):
        if temp_inf_x1.SD_rejects_H0[i] == False and np.any(P_x1[i] == 2):
            correct_with_no_match_from_CCx1_test = correct_with_no_match_from_CCx1_test + 1
        if temp_inf_x2.SD_rejects_H0[i] == False and np.any(P_x1[i] == 2):
            correct_with_no_match_from_CCx2_test = correct_with_no_match_from_CCx2_test + 1

    print(" \n Mistakes by the simple correlation method on holdout data")
    print(" Sim_Correlation  X1_train mistakes number",
          len(unmapped_features_orig) - correct_with_match_from_x1_test - num_xtra_feat_inX1, "out of ",
          len(unmapped_features_orig) - num_xtra_feat_inX1)
    print(" Sim_Correlation  X2_train mistakes number",
          len(unmapped_features_orig) - correct_with_match_from_x2_test - num_xtra_feat_inX1, "out of ",
          len(unmapped_features_orig) - num_xtra_feat_inX1)

    print("\n Mistakes by the significance testing algorithm on holdout data")
    print("From CC x1 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx1_test, " out of ", num_xtra_feat_inX1)
    print("From CC x2 ", num_xtra_feat_inX1 - correct_with_no_match_from_CCx2_test, " out of ", num_xtra_feat_inX1)


    print(" -------- Sim_Correlation  methods  ends ------------- \n \n  ")

    del df_rename_preproc

    TP_x1 = 0
    FP_x1 = 0
    TN_x1 = 0
    FN_x1 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x1_match_matrix_test[i, j] == 1):
                TP_x1 = TP_x1 + 1
            elif (P_x1[i, j] == 1) & (x1_match_matrix_test[i, j] == 0):
                FN_x1 = FN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 0):
                TN_x1 = TN_x1 + 1
            elif (P_x1[i, j] == 0) & (x1_match_matrix_test[i, j] == 1):
                FP_x1 = FP_x1 + 1

    TP_x2 = 0
    FP_x2 = 0
    TN_x2 = 0
    FN_x2 = 0
    for i in range(P_x1.shape[0]):
        for j in range(P_x1.shape[1]):
            if (P_x1[i, j] == 1) & (x2_match_matrix_test[i, j] == 1):
                TP_x2 = TP_x2 + 1
            elif (P_x1[i, j] == 1) & (x2_match_matrix_test[i, j] == 0):
                FN_x2 = FN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test[i, j] == 0):
                TN_x2 = TN_x2 + 1
            elif (P_x1[i, j] == 0) & (x2_match_matrix_test[i, j] == 1):
                FP_x2 = FP_x2 + 1
    F1_fromx1 = (2 * TP_x1) / (2 * TP_x1 + FN_x1 + FP_x1)
    F1_fromx2 = (2 * TP_x2) / (2 * TP_x2 + FN_x2 + FP_x2)



    print( "Sim cor F values ", F1_fromx1, F1_fromx2)

    return correct_with_match_from_x1_test, correct_with_match_from_x2_test, correct_with_no_match_from_CCx1_test, correct_with_no_match_from_CCx2_test, temp_inf_x1, temp_inf_x2, F1_fromx1, F1_fromx2



