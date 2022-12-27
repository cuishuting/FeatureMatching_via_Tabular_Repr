# FeatureMatching_via_Tabular_Repr
The original EHR recordings is shown as follow:
![image](https://github.com/cuishuting/FeatureMatching_via_Tabular_Repr/blob/main/IMG/original_tabular_repr.png)
A series of preprocessing operations are included in folder "Org_Data_and_Preprocess_ipynbs".
The introduction of pre-processed data files can be found in file "Processed_Data_Decription.txt"
Based on processed data we get the original tabular representation and final tabular representation after imputation,as shown below, where rows represent 24 time windows with 1 hour interval and columns represent both mapped and unmapped features in current EHR with first few columns as pre-known mapped features and left ones as unmapped features:
![image](https://github.com/cuishuting/FeatureMatching_via_Tabular_Repr/blob/main/IMG/final_tabular_repr.png)
