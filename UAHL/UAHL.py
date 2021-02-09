'''
    | ============================================================================= |
    | - UAHL {Unsupervised Analysis Framework for Heterogenous Log-Files}           |
    | - Code Developer: Ahmed Abdulrahman Alghamdi                                  |
    | - The framework is available at: https://github.com/aag1990/UAHL              |
    |   (Visit the GitHub repository for the documentation and other details)       |
    | - When using any part of this framework for research or industrial            |
    |    purposes, please cite our paper which is shown in the GitHub repository.   |
    | - Date: 25 Apr 2020                                                           |
    | ============================================================================= |
'''
import sys
import utils.utils as utils

Step1SelectedFeatures = True            # Pre-processing options: Using the selected feature sets
Phase1DataScaling = True                # Pre-processing options: Scaling the Dataframes in Phase 1
EPS_figure_plot = True                  # Results options: Shows EPS figures (k-distances curve)
ResultsDetailsPrint = True              # Results options: prints results' details on screen (Clus. and outl. details & confusion matrix)
ExportResultsToCSV = True               # Results options: Exporting results as CSV files


def main():

    print('\033[1m\n # Phase 1 - Individual log-files analysis:\033[0m')

    # Dataframes loading
    print('\033[1m\n [-] Files loading...\033[0m')
    LogFilesDataframes, fileNames, LabelsEventType, LabelsIPpairID, orgDF = utils.files_loading(sys.argv[1:])

    # Dataframes preprocessing starts
    print('\033[1m\n [-] Dataframes pre-processing %s...\033[0m' %
          ("(Each sub-dataframe will be pre-processed individually)" if len(LogFilesDataframes) > 1 else ""))

    # Joining the columns "Date" and "Time" into one column and converts them into Unix timestamps
    print('  +| Converting Date&Time columns into a single Unix-timestamps column...')
    LogFilesDataframes, DTC_ProcessDuration, orgDF = utils.DateTimeConversion(LogFilesDataframes, orgDF)

    # IP addresses formating
    print('  +| Converting IP addresses into numerical values...')
    LogFilesDataframes, IPConv_ProcessDuration, orgDF = utils.IP_addrs_formating(LogFilesDataframes, orgDF)

    # Using the selected feature sets (Applied based on the user option)
    if Step1SelectedFeatures is True:
        print('  *| Selected feature sets from the file "utils/selected_features" will be used for clustering')
        LogFilesDataframes = utils.selected_features(LogFilesDataframes, fileNames)

    # Converting categorical and text values into numerical values
    print('  +| Converting values of non-numeric columns in the data frames into numerical values')
    LogFilesDataframes, PreProc_ProcessDuration = utils.dataframes_preprocessing(LogFilesDataframes)

    # Data scaling (MinMaxScaler is used by default. The scaler can be changed in the file "utils.py")
    if Phase1DataScaling is True:
        LogFilesDataframes, scalerName, Scl_ProcessDuration = utils.dataframes_scaling(LogFilesDataframes)

    # Parameter determination (The parameter MinPts is set to 2 by default in this framework)
    print('\033[1m\n [-] Calculating the optimal value for the parameter \"EPS\"...\033[0m')
    P1_EpsList, P1_EPS_ProcessDuration = utils.parameter_determination(LogFilesDataframes, EPS_figure_plot, fileNames)

    # Events Clustering
    print('\033[1m\n [-] Applying the clustering algorithm (DBSCAN) on the dataframes...\033[0m')
    P1_pred_labels_lists, P1_ClusProcessDuration = utils.dbscan_clustering(LogFilesDataframes, P1_EpsList, 2, fileNames)

    # Results evaluation (This process is applied by default)
    print('\033[1m\n [-] Evaluating the clustering results...\033[0m')
    P1_clusters_details_List, P1_outliers_details_List, P1_hom_com_vmet_List, P1_AR_Score_List, P1_AMI_Score_List, P1_CM_List = utils.results_evaluation_phase1(fileNames, LabelsEventType, P1_pred_labels_lists, ResultsDetailsPrint)

    # Adds files' tags to the labels
    P1_pred_labels_lists = utils.P1_LabelsTags(P1_pred_labels_lists, fileNames)

    # Results exporting
    if ExportResultsToCSV is True:
        print('\033[1m\n [-] Writing results to CSV files...\033[0m')
        utils.results_exporting_phase1(orgDF, fileNames, LogFilesDataframes, LabelsEventType, LabelsIPpairID, Step1SelectedFeatures, Phase1DataScaling, P1_EpsList, 2, P1_EPS_ProcessDuration,
                                       P1_ClusProcessDuration, P1_pred_labels_lists, P1_hom_com_vmet_List, P1_AR_Score_List, P1_AMI_Score_List, P1_clusters_details_List, P1_outliers_details_List, P1_CM_List, __file__)

    print('\033[1m\n # Phase 2 - Patterns extraction over the analysed log-files:\033[0m')

    # Cross-Correlation-log generation
    print('\033[1m\n [-] Generating the Cross-Correlation-log...\033[0m')
    CCL, Phase2selectedFeatures, CCL_Proc_Duration = utils.CCL_generation(orgDF, LabelsEventType, LabelsIPpairID, P1_pred_labels_lists, fileNames)
    print('  *| %d rows were added to the CCL in %s seconds' % (len(CCL), CCL_Proc_Duration))
    print('  *| The selected set of features for the CCL is: %s' % Phase2selectedFeatures)

    # CCL preprocessing starts
    print('\033[1m\n [-] CCL dataframe pre-processing...\033[0m')
    Processed_CCL, DP_ProcessDuration = utils.dataframes_preprocessing([CCL[Phase2selectedFeatures]])  # Only selected features will be used to improve speed

    # Parameter determination
    print('\033[1m\n [-] Calculating the optimal value for the parameter \"EPS\"...\033[0m')
    P2_EpsList, P2_EPS_ProcessDuration = utils.parameter_determination(Processed_CCL, EPS_figure_plot, ['CCL'])

    # CCL Clustering (Scaling was not applied for the CCL as is breaks the meaningful links between data in the DF)
    print('\033[1m\n [-] Applying the clustering algorithm (DBSCAN) on the CCL...\033[0m')
    print('  *| Only selected features will be used for the clustering process')
    P2_pred_labels_lists, P2_ClusProcessDuration = utils.dbscan_clustering(Processed_CCL, P2_EpsList, 2, ['CCL'])

    # Results evaluation
    print('\033[1m\n [-] Evaluating the CCL clustering results (this might take some time)...\033[0m')
    P2_clusters_details, P2_outliers_details, P2_hom_com_vmet, P2_AR_Score, P2_AMI_Score = utils.results_evaluation_phase2(CCL['label_IPpair_ID~'].values.tolist(), P2_pred_labels_lists[0])

    # Results exporting
    if ExportResultsToCSV is True:
        print('\033[1m\n [-] Writing the CCL results to a CSV file...\033[0m')
        utils.results_exporting_P2(CCL, Phase2selectedFeatures, P2_EpsList, 2, P2_EPS_ProcessDuration, P2_ClusProcessDuration,
                                   P2_pred_labels_lists, P2_hom_com_vmet, P2_AR_Score, P2_AMI_Score, P2_clusters_details, P2_outliers_details, __file__)

# Runs the script
if __name__ == '__main__':
    main()
