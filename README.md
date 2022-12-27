# FeatureMatching_via_Tabular_Repr
The original EHR recordings is shown as follow:
![image](https://github.com/cuishuting/FeatureMatching_via_Tabular_Repr/blob/main/IMG/original_tabular_repr.png)
A series of preprocessing operations are included in folder "Org_Data_and_Preprocess_ipynbs".
The introduction of pre-processed data files can be found in file "Processed_Data_Decription.txt"
Based on processed data we get the original tabular representation and final tabular representation after imputation,as shown below, where rows represent 24 time windows with 1 hour interval and columns represent both mapped and unmapped features in current EHR with first few columns as pre-known mapped features and left ones as unmapped features:
![image](https://github.com/cuishuting/FeatureMatching_via_Tabular_Repr/blob/main/IMG/final_tabular_repr.png)
Concatenating all patients' final tabular representations we get the fingerprint matrix for both mapped and unmapped features. Then we applied KMF then Gale-Shapley algorithm to get predicted matched pairs.
![image](https://github.com/cuishuting/FeatureMatching_via_Tabular_Repr/blob/main/IMG/KMF_match.png)

The feature matching result via tabular representation is with F-1 score 0.466. Details of matching are shown below.
![image](https://github.com/cuishuting/FeatureMatching_via_Tabular_Repr/blob/main/IMG/tab_repr_result_1.png)
![image](https://github.com/cuishuting/FeatureMatching_via_Tabular_Repr/blob/main/IMG/tab_repr_result_2.png)
