'''
    | ============================================================================= |
    | - UAHL {Unsupervised Analysis Framework for Heterogenous Log-Files}           |
    | - Code Developer: Ahmed Abdulrahman Alghamdi                                  |
    | - The framework is available at: https://github.com/aag1990/UAHL              |
    |   (Visit the GitHub repository for the documentation and other details)       |
    | - When using any part of this framework for research or industrial            |
    |    purposes. Please cite our paper which is shown in the GitHub repository.   |
    | - Date: 25 Apr 2020                                                           |
    | ============================================================================= |
'''
import os, sys, re, warnings, csv, re
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from prettytable import PrettyTable
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, homogeneity_completeness_v_measure, adjusted_mutual_info_score, adjusted_rand_score

# Imports the lists of selected features for each log-file
currentPath = os.path.dirname(os.path.realpath(__file__))
selected_features_list = eval(open(str(currentPath) + "/selected_features").read())


def files_loading(args):
    # Containers for the imported log-files, the separated labels, the file names, and copy of the files
    LogFilesDataframes, fileNames, LabelsEventType, LabelsIPpairID, orgDF = [], [], [], [], []
    for i in range(len(args)): 
        try:
            log_file = pd.read_csv(args[i], sep=",", comment='#', low_memory=False)     # Imports the log-file
            LabelsEventType.append(log_file['label_event_type~'].values.tolist())       # Separates labels into a list
            LabelsIPpairID.append(log_file['label_IPpair_ID~'].values.tolist())         # Separates labels into a list

            LogFilesDataframes.append(log_file[log_file.columns.difference(['label_event_type~', 'label_IPpair_ID~'])]) # Appends the imported log-file to the DF list (labels in the DF were dropped)
            fileNames.append(os.path.splitext(os.path.basename(args[i]))[0])    # Appends the file name
            orgDF.append(log_file)                                              # Makes a copy of the DF (To be used when exporting results)
            print('  *| Imported log-file: "%s"  (%s instances, %s columns(without labels), %s unique labels)' %
                  (os.path.basename(args[i]), len(log_file), len(LogFilesDataframes[-1].columns.tolist()), len(set(LabelsEventType[-1]))))
        except pd.errors.ParserError:
            pass       # Ignores non-CSV files

    print('  *| %s log-files (CSV) are imported into the framework' %len(LogFilesDataframes))
    return LogFilesDataframes, fileNames, LabelsEventType, LabelsIPpairID, orgDF


def DateTimeConversion(LogFilesDataframes, orgDF):
    start_time = datetime.now()
    for DS in range(len(LogFilesDataframes)):   # (Each dataframe is processed individually)
        col = ['DateTime~'] + list(set(LogFilesDataframes[DS].columns.values.tolist()) - {'Date~', 'Time~'})
        # Converts rows to Unix-timestamps (GMT)
        ConvLst = pd.to_datetime(LogFilesDataframes[DS]["Date~"] + " " + LogFilesDataframes[DS]["Time~"]).astype(int) // 10**9
        # Combines Date&Time columns into the new column "DateTime~"
        LogFilesDataframes[DS].insert(1, 'DateTime~', ConvLst)
        # Drops the old Date&Time columns
        LogFilesDataframes[DS]=LogFilesDataframes[DS].drop(['Date~', 'Time~'], axis=1)
        LogFilesDataframes[DS] = LogFilesDataframes[DS].reindex(columns=col)     # Re-index the dataframe's columns

        orgDF[DS] = LogFilesDataframes[DS]  # Applies the datetime modifications on the orgDF
    print('  *| Date&Time columns in the dataframe were combined into a single Unix-timestamp column in %f seconds' %
          (datetime.now() - start_time).total_seconds())
    return LogFilesDataframes, (datetime.now() - start_time).total_seconds(), orgDF


def IP_addrs_formating(LogFilesDataframes, orgDF):
    start_time = datetime.now()
    for DS in range(len(LogFilesDataframes)):   # (Each dataframe is processed individually)
        IP_addrs_columns = [i for i in LogFilesDataframes[DS].columns.tolist() if str(i)[-1] is "$"]    # List of IP addresses columns
        for c in IP_addrs_columns:                                                      # Processes columns containing IP addresses
            unique_IP_addrs = LogFilesDataframes[DS][c].unique()                        # Extracts unique IP addresses
            LogFilesDataframes[DS][c] = LogFilesDataframes[DS][c].replace("-", "9000")  # Replaces the "-" values in the column with "9000"
            for ip in unique_IP_addrs:
                if ip != "-":
                    # Formating the IP address (e.g 69.107.121.41 -> 069107121041)
                    new_ip = "9%s" % "".join([('{0: >3}'.format(s)).replace(" ", "0") for s in ip.split(".")])   
                    if new_ip == "9000":    new_ip = 9000000000000
                    # Replace IP addresses format in the column
                    LogFilesDataframes[DS][c] = LogFilesDataframes[DS][c].replace(ip, str(new_ip))

            # Converts the column type to numerical
            LogFilesDataframes[DS][c] = LogFilesDataframes[DS][c].apply(pd.to_numeric)

    orgDF[DS] = LogFilesDataframes[DS]          # Applies the IPaddresses modifications on the orgDF
    print('  *| All IP addresses were converted into numerical values in %f seconds' %(datetime.now() - start_time).total_seconds())
    return LogFilesDataframes, (datetime.now() - start_time).total_seconds(), orgDF


def selected_features(LogFilesDataframes, fileNames):
    for DS in range(len(LogFilesDataframes)):   # (Each dataframe is processed individually)
        LogFilesDataframes[DS] = LogFilesDataframes[DS][selected_features_list[fileNames[DS]]]  # Applies the selected feature sets
    return LogFilesDataframes


def dataframes_preprocessing(LogFilesDataframes):
    start_time = datetime.now()

    for DS in range(len(LogFilesDataframes)):   # (Each dataframe is processed individually)
        ListOfColumns = LogFilesDataframes[DS].select_dtypes(object).columns.tolist()
        OrdColumns = [col for col in ListOfColumns if re.search(r'~$', col)]            # ordinal categorical columns (Ends with ~)
        TextColumns = [col for col in ListOfColumns if re.search(r'@$', col)]           # Text columns (Ends with @)
        IPaddColumns = [col for col in ListOfColumns if re.search(r'\$$', col)]         # IP addresses columns (Ends with $)
        NomColumns =  [col for col in ListOfColumns if re.search(r'[^~|@|\$]$', col)]   # nominal categorical columns

        if len(OrdColumns) > 0:
            # Using LabelEncoder to convert values of the ordinal categorical columns into numerical values
            LogFilesDataframes[DS] = OrdColum_values_conversion(LogFilesDataframes[DS], OrdColumns)

        if len(NomColumns) > 0:
            # Using OneHotEncoder to convert values of the nominal categorical columns into numerical values
            LogFilesDataframes[DS] = NomColum_values_conversion(LogFilesDataframes[DS], NomColumns)

        if len(TextColumns) > 0:
            # Converting values the text columns into numerical values
            LogFilesDataframes[DS] = text_to_numerical(LogFilesDataframes[DS], TextColumns)

    print('  *| The data converting step was performed in %f seconds' %(datetime.now() - start_time).total_seconds())
    return LogFilesDataframes, (datetime.now() - start_time).total_seconds()


def NomColum_values_conversion(dataset, NomColumns):
    # Create dummies for the selected features
    dataset = pd.get_dummies(dataset, columns=NomColumns)
    return dataset


def OrdColum_values_conversion(dataset, OrdColumns):
    # Inserts the categorical columns into a 2D numpy array
    dataset_categorical_values = dataset[OrdColumns]
    dataset_categorical_values_enc = dataset_categorical_values.apply(LabelEncoder(
    ).fit_transform)  # Transforms categorical values into numbers using LabelEncoder()
    # Joins the non-categorical features and the encoded features into a new dataframe
    dataset = pd.concat([dataset_categorical_values_enc, dataset[dataset.columns.difference(OrdColumns)]], axis=1)
    return dataset


def text_to_numerical(dataset, text_columns):
    for c in text_columns:
        new_strings = []
        unique_values = []  # This list holds the unique words in the strings
        for i in range(len(dataset[c])):
            new_strings.append(dataset[c].iloc[i])
            new_strings[-1] = tsplit(new_strings[-1])   # Splits the text using regex
            unique_values.extend(set(new_strings[-1]))  # Extracts the unique values in the strings

        unique_values = sorted(set(unique_values))      # Finds unique values for all splitted strings
        labelledUniqueValues = LabelEncoder().fit_transform(unique_values)   # Encodes the unique strings into numerical values
        MaxLength = len(str(labelledUniqueValues[-1]))  # Finds the last number's length to be used as a standard length for all numbers
        dictionary = dict(zip(unique_values, labelledUniqueValues))     # Creates dictionary from the two lists (unique_values and labelledUniqueValues)

        for i in range(len(new_strings)):
            for ii in range(len(new_strings[i])):
                new_strings[i][ii] = '{0: >{prec}}'.format(dictionary[new_strings[i][ii]], prec=MaxLength).replace(" ", "0")   # Converts the original strings into numbers
            new_strings[i] = int('9%s' %''.join(new_strings[i]))   # Joins the line strings to construct the final value for the line

        MaxLength2 = len(str(max(new_strings)))
        for l in range(len(new_strings)):
            # if new_strings[i] != "-":
            new_strings[l] = '{0: <{prec}}'.format(new_strings[l], prec=MaxLength2).replace(" ", "0")   # Converts the original strings into numbers

        dataset[c] = np.asarray(new_strings)   # Includes the values into the dataframe
        dataset[c] = dataset[c].astype(float)

    return dataset


def tsplit(s):
    stack = re.split(r'(\/|\.|\(|\)|\:|\#|\=|\s|\_|\-|\[|\])', s)   # Splits the string using regex
    stack = ' '.join(stack).split()                                 # Removes None items from the list
    return stack


def dataframes_scaling(LogFilesDataframes):
    scaler = MinMaxScaler(feature_range=(0, 1))
    print('  +| Dataframes scaling using the scaler: "%s"...'%scaler.__class__.__name__)
    start_time = datetime.now()
    for DS in range(len(LogFilesDataframes)):   # (Each dataframe is processed individually)
        # Saves the column names so it could be used later to rebuild the dataframe after scaling
        dataframeColumns = LogFilesDataframes[DS].columns
        # Converts data into float data type
        LogFilesDataframes[DS] = LogFilesDataframes[DS].astype(float)               # Changes the data type to float
        LogFilesDataframes[DS] = scaler.fit_transform(LogFilesDataframes[DS])       # Scales the dataframe
        # Converts back the resulting list into a dataframe
        LogFilesDataframes[DS] = pd.DataFrame(LogFilesDataframes[DS], columns=dataframeColumns)

    global scalerName
    scalerName = scaler.__class__.__name__

    print('  *| The scaling step for the dataframes was performed in %f seconds' %((datetime.now() - start_time).total_seconds()))
    return LogFilesDataframes, scaler.__class__.__name__, (datetime.now() - start_time).total_seconds()


def parameter_determination(LogFilesDataframes, plot_graph, fileNames):
    EPS_Values = []     # Container for the calculated EPS values of the dataframe
    AllProcDuration = []
    for DS in range(len(LogFilesDataframes)):   # (Each dataframe is processed individually)
        start_time = datetime.now()
        # Initiates the parameter variables, MinPts is set to 2 by default
        Eps, MinPts = None, 2
        # Extracts unique datapoints (this will minimise errors and processing time, also removes the zero slopes)
        dataset = LogFilesDataframes[DS].drop_duplicates()

        if len(dataset) is 1:  # When all datapoints are the same (Duplicated datapoints)
            print(' *| No distances calculated between datapoints (Duplicated datapoints), thus all datapoints will be grouped in one cluster')
            EPS_Values.append(0)

        else:
            # Extracts distances between datapoints by using KNN
            neigh = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(dataset)   # You could change the algorithm to "brute"
            # Distances is a 2d-list which contains distances between points
            distances, indices = neigh.kneighbors(dataset)
            # Sorts the second column of the distances list (first column is the distance between a point and itself)
            distances = np.sort(distances[:, 1])

            if len(set(distances)) is 1:    # When distances between datapoints are the same (usually when dealing with 0 and 1 data)
                print('  *| "{}": All calculated non-zero slopes between the datapoints are the same. EPS is set to {} (distances[0]/2)' .format(fileNames[DS], (distances[0] / 2)))
                # Returns half of the only value (distance) exists in the distances list (this will separate the clusters correctly)
                EPS_Values.append(distances[0] / 2)
                AllProcDuration.append((datetime.now() - start_time).total_seconds())

            else:
                # Set the EPS value to the mean of unique calculated distances
                Eps = np.mean(np.unique(distances))
                print("  *| The optimal EPS parameter value for the dataframe \"%s\" is: %s" %(fileNames[DS], Eps))
                EPS_Values.append(Eps)
                AllProcDuration.append((datetime.now() - start_time).total_seconds())

                # Shows the plot
                if plot_graph is True:  
                    plt.plot(distances)                     # Line1: Distances
                    plt.plot(([Eps] * len(distances)))      # Line2: EPS value
                    # Finds the intersecting point (EPS value)
                    intersection = interp1d(distances-([Eps] * len(distances)), np.arange(distances.shape[0]))(0)
                    plt.scatter(intersection, Eps, label=("Automatically calculated EPS={}" .format(
                        Eps)), c='red', s=40, marker="o")   # Marks the selected EPS value
                    plt.xlabel("Datapoints")
                    plt.ylabel("k-Distances")
                    plt.title(fileNames[DS], fontsize=10, y=1.02, fontweight='bold')
                    plt.legend(loc='upper left')
                    plt.grid(linestyle='dotted')
                    plt.show()

    print('  *| The parameters calculation was performed in %f seconds' %sum(AllProcDuration))
    return EPS_Values, AllProcDuration


def dbscan_clustering(LogFilesDataframes, EpsList, MinPts, fileNames):
    PredictedLabelsList, AllProcDuration  = [], []          # Containers for resulting predicted labels and processing durations
    for DS in range(len(LogFilesDataframes)):               # (Each dataframe is processed individually)
        start_time = datetime.now()
        if EpsList[DS] == 0:    # When all datapoints are duplicated (no distances between them)
            print('  *| DBSCAN was not applied on the dataframe %s as there are no distances between datapoints' % fileNames[DS])
            print('          All datapoints were grouped into one cluster')
            pred_labels = [0] * len(LogFilesDataframes[DS])
            PredictedLabelsList.append(list(pred_labels))   # Appends results
        else:
            # Applies the DBSCAN only when Eps is a valid value (>0.0)
            dbscan = DBSCAN(eps=EpsList[DS], min_samples=MinPts, metric="euclidean")
            cls = dbscan.fit(LogFilesDataframes[DS])  # Applies the clustering algorithm on data
            pred_labels = cls.labels_
            clustering_process_duration = (datetime.now() - start_time).total_seconds()
            print('  *| Data clustering is performed on the dataframe \"%s\" in %f seconds' %(fileNames[DS], clustering_process_duration))
            PredictedLabelsList.append(list(pred_labels))
            AllProcDuration.append(clustering_process_duration)

    if len(LogFilesDataframes) > 1:
        print('  *| Total duration of the dataframes\' clustering process is %s seconds' %sum(AllProcDuration))
    return PredictedLabelsList, AllProcDuration


def P1_LabelsTags(pred_labels_lists, fileNames):
    for DS in range(len(fileNames)):   # (Each dataframe is processed individually)
        # Converts labelling numbers to strings by adding letters for each logfile (e.g.: 50 -> A50)
        FileTag = selected_features_list['labelling_' + fileNames[DS]]
        pred_labels_lists[DS] = [(FileTag + '_' + str(i)) if i!=-1 else (FileTag + '_OUTLIER') for i in pred_labels_lists[DS]]
    return pred_labels_lists


def results_evaluation_phase1(fileNames, actual_labelsList, predicted_labelsList, ResultsDetailsPrint):
    clusters_details_List, outliers_details_List, hom_com_vmet_List, AR_Score_List, AMI_Score_List, CM_List = [], [], [], [], [], []

    for DS in range(len(fileNames)):   # (Each dataframe is processed individually)
        # Combines the two lists into a DF thus Pandas inquiries can be used on them
        actual_vs_predicted_labels_df = pd.DataFrame({'actual_labels': actual_labelsList[DS], 'predicted_labels': predicted_labelsList[DS]})

        clusters_details = []    # This list includes details of each clusters (majority voting, Number of items, items distribution)
        clusters_MV_labels = {}  # A dictionary includes each cluster's concluded label found by the majority voting

        # Extracts clusters' details            
        for c in [x for x in sorted(set(predicted_labelsList[DS])) if x >= 0]:                  # Evaluates each cluster (except outliers)
            details = dict(actual_vs_predicted_labels_df.query("predicted_labels == @c ")[
                           'actual_labels'].value_counts())                                     # Counts actual labels inside the cluster
            details = {i: details[i] for i in sorted(details.keys())}                           # Sorts directory by keys
            # Applies the majority voting to extract the cluster's label
            clusters_MV_labels.update({c: (max(details, key=details.get)) + str(c)})
            clusters_details.append('       # Cluster [%s] contains %d items. MV:[\"%s\"]. Details:%s' % (str(
                c), list(predicted_labelsList[DS]).count(c), clusters_MV_labels[c], details))   # Adds the cluster's details to the list
        clusters_MV_labels.update({-1: -1})                                                     # Keeps the outliers without MV

        # Extracts outliers details
        if -1 in actual_vs_predicted_labels_df["predicted_labels"].tolist():
            outliers_details = dict(actual_vs_predicted_labels_df.query("predicted_labels == -1")['actual_labels'].value_counts())
            outliers_details = dict(sorted(outliers_details.items(), key=lambda x: x[0]))
        else:
            outliers_details = ""

        actual_vs_predicted_labels_df['predicted_labels'] = actual_vs_predicted_labels_df['predicted_labels'].map(clusters_MV_labels)  # pred. labels replaced with the MV

        # Calculates homogeneity, completeness, Vmeasure, AR, and AMI scores
        warnings.filterwarnings('ignore')   # Ignores warning messages generated when there are labels in the actual labels list, but not shown in the "predicted_labels" list. The sklearn would handle this by replacing them with 0. THIS WOULD NOT CHANGE THE RESULTS. Howewer, we wanted to hide the warning message on the screen.
        MetricWithoutOtl = actual_vs_predicted_labels_df[actual_vs_predicted_labels_df['predicted_labels'] != -1]  # Excludes outliers
        hom_com_vmet_List.append(homogeneity_completeness_v_measure(
            MetricWithoutOtl['actual_labels'], MetricWithoutOtl['predicted_labels']))
        AR_Score_List.append(adjusted_rand_score(MetricWithoutOtl['actual_labels'], MetricWithoutOtl['predicted_labels']))
        AMI_Score_List.append(adjusted_mutual_info_score(MetricWithoutOtl['actual_labels'], MetricWithoutOtl['predicted_labels']))
        CM_List.append(conf_matrix_print(
            MetricWithoutOtl['actual_labels'].values, MetricWithoutOtl['predicted_labels'].values))  # Generates the confusion matrix

        # Prints the results' summary
        print('  *| File "{}" results: ({} Clus. | {} Outl. | Homg.:{:.2%} | Comp.:{:.2%} | V-measure:{:.2%} | AR:{:.2%} | AMI:{:.2%})' .format(fileNames[DS], len(MetricWithoutOtl['predicted_labels'].unique()), list(predicted_labelsList[DS]).count(-1), hom_com_vmet_List[-1][0], hom_com_vmet_List[-1][1], hom_com_vmet_List[-1][2], AR_Score_List[-1], AMI_Score_List[-1]))

        # Prints the clusters and outliers details
        if ResultsDetailsPrint is True:
            if -1 in actual_vs_predicted_labels_df["predicted_labels"].tolist():
                print('     - Outliers list contains: %s' % outliers_details)
            print('     - Details of the resulting clusters:')
            print('\n' .join(clusters_details))
            print(CM_List[-1])

        clusters_details_List.append(clusters_details)
        outliers_details_List.append(outliers_details)
    return clusters_details_List, outliers_details_List, hom_com_vmet_List, AR_Score_List, AMI_Score_List, CM_List


def conf_matrix_print(actual_labels, predicted_labels):
    predicted_labels = [(''.join([i for i in d if not i.isdigit()])) for d in predicted_labels]
    SortedUniqueLabelsList = sorted(set(actual_labels))                                                     # Extracts unique label names
    conf_matrix = confusion_matrix(actual_labels, predicted_labels, labels=SortedUniqueLabelsList).tolist() # Builds the confusion matrix

    # Removes 0 values in the confusion matrix, thus results can be easier to read by users
    for l in range(len(conf_matrix)):
        for ll in range(len(conf_matrix[l])):
            if conf_matrix[l][ll] == 0: conf_matrix[l][ll] = ""

    x = PrettyTable()
    if len(SortedUniqueLabelsList) > 3:                 # Table formating:
        for l in range(len(SortedUniqueLabelsList)):    # Adjusts label lengths in the table (when labels are more than 3)
            if (4 - len(SortedUniqueLabelsList[l])) > 0:
                SortedUniqueLabelsList[l] = SortedUniqueLabelsList[l].center(4, " ")    # Centering strings in cells
    else:
        for l in range(len(SortedUniqueLabelsList)):    # Adjusts label lengths in the table (when labels are less than 3)
            if (9 - len(SortedUniqueLabelsList[l])) > 0:
                SortedUniqueLabelsList[l] = SortedUniqueLabelsList[l].center(9, " ")    # Centering strings in cells
    TableHeader = ""

    if len(SortedUniqueLabelsList) > 3:                 # Adjusts the table's header when labels are more than 3
        TableHeader = ('       +--------+' + (('-------' * len(SortedUniqueLabelsList))[:-1]) + '+\n       | Actual +' + ("Predicted".center((len(SortedUniqueLabelsList) * 7), ' '))[1:] + '|\n')
    else:
        # Adjusts the table's header when labels are less than 3
        TableHeader = ('      +-----------+' + (('------------' * len(SortedUniqueLabelsList))[:-1]) + '+\n      |  Actual   +' + (" Predicted".center((len(SortedUniqueLabelsList) * 12), ' '))[1:] + '|\n')

    # Table content
    column_names = ['      '] + SortedUniqueLabelsList     # Sets the second row of the table, which includes the predicted labels
    x.add_column(column_names[0], SortedUniqueLabelsList)  # Adds actual_labels columns
    for c in range(len(SortedUniqueLabelsList)): x.add_column(column_names[c + 1], [item[c] for item in conf_matrix])  # Adds results to CM
    full_table = str(TableHeader) + "       "  + (str(x).replace("\n", "\n       "))    # The table header & The confusion matrix table
    return full_table


def results_exporting_phase1(orgDF, fileNames, LogFilesDataframes, LabelsEventType, LabelsIPpairID, SelectedFeatures, Phase1DataScaling, EpsList, MinPts, EPS_ProcessDuration, ClusProcessDuration, pred_labels_lists, hom_com_vmet_List, AR_Score_List, AMI_Score_List, clusters_details_List, outliers_details_List, CM_List, directory):

    global NewFolderName
    path = os.path.dirname(directory)  # Extracts the current script path
    NewFolderName = "%s/Results-%s" % (path, datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(NewFolderName, exist_ok=True)  # Creates new folder for results
    os.makedirs("%s/Phase1_Results/" % (NewFolderName), exist_ok=True)  # Creates new folder for Phase1 results

    for DS in range(len(fileNames)):   # (Each dataframe is processed individually)
        # Constructs the file header
        File_Header = ("#" + ("-" * 108) + "\n")
        File_Header+= ("#  ~ DateTime: %s\n" %datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        File_Header+= ("#  ~ Dataframe name: \"%s\"\n" % fileNames[DS])
        File_Header+= ("#  ~ Number of events in the dataframe: %d\n" %len(LogFilesDataframes[DS]))
        File_Header+= ("#  ~ Number of unique labels ('label_event_type') in the dataframe: %s\n" %len(set(LabelsEventType[DS])))

        if SelectedFeatures is True: 
            File_Header += ("#  ~ Selected features set for the clustering: %s\n" %selected_features_list[fileNames[DS]])
        else: File_Header += ("#  ~ All columns in the dataframe (excluding the labels) were used for the clustering\n")

        if Phase1DataScaling is True: File_Header+= ("#  ~ The dataframe was scaled using the scaler: %s\n" %scalerName)

        if EpsList[DS] != 0: File_Header+= ("#  ~ Parameters:  EPS={}  MinPts={} (The EPS parameter was calculated in {} seconds)\n" .format(EpsList[DS], MinPts, EPS_ProcessDuration[DS]))
        else:   File_Header+= ("#  ~ Calculated parameters:  An error occurred while performing this process (EPS was set to 0)")

        File_Header += ("#  ~ The DBSCAN clustering process was performed in %f seconds" %ClusProcessDuration[DS])

        File_Header += ("\n#  ~ Results summary: ({} Clus. | {} Outl. | Homg.: {:.2%} | Comp.: {:.2%} | V-measure: {:.2%} | AR: {:.2%} | AMI: {:.2%})" .format(len([x for x in set(pred_labels_lists[DS]) if 'OUTLIER' not in x]), len([x for x in pred_labels_lists[DS] if "OUTLIER" in x]), hom_com_vmet_List[DS][0], hom_com_vmet_List[DS][1], hom_com_vmet_List[DS][2], AR_Score_List[DS], AMI_Score_List[DS]))

        File_Header += ("\n#\n#  ~ Details of clusters:\n#       %s" %"\n#       " .join(i[9:] for i in clusters_details_List[DS]))
        if outliers_details_List[DS] != '': File_Header+= ("\n#\n#  ~ Details of outliers: %s" %outliers_details_List[DS])

        File_Header += ("\n#\n#  ~ Confusion matrix:\n")
        File_Header += ("#" + str(CM_List[DS][2:]).replace("\n  ", "\n#").replace("#|", "# "))    # Styling the CM

        File_Header += ("\n#\n#  ~ Note: to import this file into Python, use: \"DF = pandas.read_csv(file_name, sep=',', comment='#')\"")
        File_Header += ("\n#\n#" + ("-" * 108) + "\n")

        orgDF[DS]['label_event_type~'] = LabelsEventType[DS]                 # Re-cobmine labels
        orgDF[DS]['label_IPpair_ID~'] = LabelsIPpairID[DS]                   # Re-cobmine labels
        orgDF[DS]['phase1_predicted_label~'] = pred_labels_lists[DS]         # Re-combine predected labels with the DF

        ExportedFileName = ("%s/Phase1_Results/Res_%s.csv" %(NewFolderName, fileNames[DS]))     # Sets filename
        file = open(ExportedFileName, "w")                                                      # Creates a new file to write the DF
        file.write(str(File_Header))                                                            # Writes the file header first
        file.write(str(',' .join(x for x in orgDF[DS].columns))+'\n')                           # Writes the DS columns at the first line
        orgDF[DS].to_csv(file, index=None, header=False, quoting=csv.QUOTE_NONNUMERIC)          # writes the dataset's rows
        print('  *| Clustering results for the dataframe "%s" exported to the file: %s' %(fileNames[DS], ExportedFileName))


def CCL_generation(dataframes, LabelsEventType, LabelsIPpairID, pred_labels_lists, fileNames):

    start_time = datetime.now()
    columns_set = ['DateTime~', 'LoggingDevice', 'Logging_Daemon', 'StatusCode', 'EUID', 'Protocol',
                   'RuleNumber', 'PID', 'User~', 'SrcIP$', 'SrcPort~', 'DstIP$', 'DstPort~', 'Message@',
                   'label_event_type~', 'label_IPpair_ID~', 'phase1_predicted_label~', 'FileName', 'Org_index'] # Set of common features
    CCL = pd.DataFrame(columns=columns_set)

    print('  +| Inserting all rows of the log-files into the CCL...')
    for L in range(len(dataframes)):   # (Each dataframe is processed individually)

        # Changes the columns names so they could match with the CCL columns names
        if len(selected_features_list[('Res_'+fileNames[L])].keys()) > 0:
            for k in selected_features_list[('Res_'+fileNames[L])].keys():
                dataframes[L] = dataframes[L].rename(columns={k: selected_features_list[('Res_'+fileNames[L])][k]})
            # Combines the Message and Message2 columns to get enough data in the Message column
            if 'Message2' in dataframes[L].columns.tolist():
                dataframes[L]['Message@'] = dataframes[L]['Message'] + ' : ' + dataframes[L]['Message2']
                dataframes[L]['Message@'] = [re.sub('^\- \: | \: \-$', "", s) for s in dataframes[L]['Message@'].values.tolist()]  # removes extra punctuations (if exist) from the begining of the text

        dataframes[L]['FileName'] = fileNames[L]                                # Inserts the file name as a new column
        dataframes[L]['Org_index'] = dataframes[L].index.tolist()               # Inserts the rows index as a new column
        dataframes[L]['label_event_type~'] = LabelsEventType[L]                 # Re-cobmine labels
        dataframes[L]['label_IPpair_ID~'] = LabelsIPpairID[L]                   # Re-cobmine labels
        dataframes[L]['phase1_predicted_label~'] = pred_labels_lists[L]         # Re-combine predected labels

        ComLis = list(set(CCL.columns.tolist()) & set(dataframes[L].columns.tolist()))          # List of common features
        CCL = CCL.append(dataframes[L][ComLis], ignore_index=True, sort=True).fillna("-")       # Inserts the DataFrame into the CCL

    # Replaces the "-" values in the columns with "0"
    CCL[['PID', 'SrcIP$', 'DstIP$','SrcPort~', 'DstPort~']] = CCL[['PID', 'SrcIP$', 'DstIP$','SrcPort~', 'DstPort~']].replace("-", 0)

    CCL = CCL.reindex(columns=columns_set)   # Re-order the columns' names
    print('  +| Sorting rows of the CCL...')
    CCL = CCL.sort_values(['DateTime~'])
    return CCL, selected_features_list['Phase2_SelectedFeatures'], (datetime.now() - start_time).total_seconds()


def results_evaluation_phase2(actual_labels, predicted_labels):
    start_time = datetime.now()

    print('  +| Extracting details of the resulting clusters...')
    actual_vs_predicted_labels_df = pd.DataFrame({'actual_labels': actual_labels, 'predicted_labels': predicted_labels})
    clusters_details = []    # The list includes details of each clusters (Number of items & items distribution)
    # Extracts clusters' details
    for c in [x for x in sorted(set(predicted_labels)) if x >= 0]:      # Evaluates each cluster (except outliers)
        details = dict(actual_vs_predicted_labels_df.query("predicted_labels == @c ")[
            'actual_labels'].value_counts())                                     # Counts actual labels inside the cluster

        details = {i: details[i] for i in sorted(details.keys())}                           # Sorts directory by keys
        clusters_details.append('       # Cluster [%s] contains %d items. Details:%s' % (str(
            c), list(predicted_labels).count(c), details))   # Adds the cluster's details to the list

    # Extracts outliers details
    if -1 in actual_vs_predicted_labels_df["predicted_labels"].tolist():
        outliers_details = dict(actual_vs_predicted_labels_df.query("predicted_labels == -1")['actual_labels'].value_counts())
        outliers_details = dict(sorted(outliers_details.items(), key=lambda x: x[0]))
    else:
        outliers_details = ""

    # Calculates homogeneity, completeness, Vmeasure, AR, and AMI scores
    warnings.filterwarnings('ignore')   # Ignores outliers
    print('  +| Calculating the clustering evaluation metrics...')
    MetricWithoutOtl = actual_vs_predicted_labels_df[actual_vs_predicted_labels_df['predicted_labels'] != -1]
    P2_hom_com_vmet = (homogeneity_completeness_v_measure(MetricWithoutOtl['actual_labels'], MetricWithoutOtl['predicted_labels']))
    P2_AR_Score = (adjusted_rand_score(MetricWithoutOtl['actual_labels'], MetricWithoutOtl['predicted_labels']))
    P2_AMI_Score = (adjusted_mutual_info_score(MetricWithoutOtl['actual_labels'], MetricWithoutOtl['predicted_labels']))

    # Prints the results' summary
    print('  *| Summary of Phase2 clustering results: ({} Clus. | {} Outl. | Homg.:{:.2%} | Comp.:{:.2%} | V-measure:{:.2%} | AR:{:.2%} | AMI:{:.2%})' .format(len(MetricWithoutOtl['predicted_labels'].unique()), list(predicted_labels).count(-1), P2_hom_com_vmet[0], P2_hom_com_vmet[1], P2_hom_com_vmet[2], P2_AR_Score, P2_AMI_Score))
    return clusters_details, outliers_details, P2_hom_com_vmet, P2_AR_Score, P2_AMI_Score


def results_exporting_P2(CCL, Phase2selectedFeatures, P2_EpsList, P2_MinPts, P2_EPS_ProcessDuration, P2_ClusProcessDuration, P2_pred_labels, P2_hom_com_vmet, P2_AR_Score, P2_AMI_Score, P2_clusters_details, P2_outliers_details, directory):
    # Constructs the file header
    File_Header = ("#" + ("-" * 108) + "\n")
    File_Header += ("#  ~ DateTime: %s\n" %datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    File_Header += ("#  ~ Dataframe name: \"CCL\"\n")
    File_Header += ("#  ~ Number of events in the dataframe: %d\n" %len(CCL))
    File_Header += ("#  ~ Number of unique IP pairs ('label_IPpair_ID') in the dataframe: %s\n" %len(CCL['label_IPpair_ID~'].unique()))
    File_Header += ("#  ~ Selected features set for the clustering: %s\n" %Phase2selectedFeatures)
    File_Header += ("#  ~ No scaler was used for the CCL dataframe\n")
        
    if P2_EpsList[0] != 0:
        File_Header += ("#  ~ Parameters:  EPS={}  MinPts={} (The EPS parameter was calculated in {} seconds)\n" .format(
            P2_EpsList[0], P2_MinPts, P2_EPS_ProcessDuration[0]))
    else:
        File_Header += ("#  ~ Calculated parameters:  An error occurred while performing this process (EPS was set to 0)")

    File_Header += ("#  ~ The DBSCAN clustering process was performed in %f seconds" %P2_ClusProcessDuration[0])

    File_Header += ("\n#  ~ Results summary: ({} Clus. | {} Outl. | Homg.: {:.2%} | Comp.: {:.2%} | V-measure: {:.2%} | AR: {:.2%} | AMI: {:.2%})" .format(len([x for x in set(P2_pred_labels[0]) if x != -1]), list(P2_pred_labels[0]).count(-1), P2_hom_com_vmet[0], P2_hom_com_vmet[1], P2_hom_com_vmet[2], P2_AR_Score, P2_AMI_Score))

    File_Header += ("\n#\n#  ~ Details of clusters:\n#       %s" %"\n#       " .join(i[9:] for i in P2_clusters_details))
    if P2_outliers_details != '':
        File_Header += ("\n#\n#  ~ Details of outliers: %s" %P2_outliers_details)

    File_Header += ("\n#\n#  ~ Note: to import this file into Python, use: \"DF = pandas.read_csv(file_name, sep=',', comment='#')\"")
    File_Header += ("\n#\n#" + ("-" * 108) + "\n")

    # Includes the clustering results with the original dataset as a new column
    dataset = CCL.assign(phase2_predicted_label=P2_pred_labels[0])

    os.makedirs("%s/Phase2_Results" % (NewFolderName), exist_ok=True)               # Creates new folder for the CCL results
    ExportedFileName = ("%s/Phase2_Results/Res_CCL.csv" %NewFolderName)             # Sets filename
    file = open(ExportedFileName, "w")                                              # Creates a new file to write the DF
    file.write(str(File_Header))                                                    # Writes the file header first
    file.write(str(',' .join(x for x in dataset.columns))+'\n')                     # Writes the dataset's columns at the first line
    dataset.to_csv(file, index=None, header=False, quoting=csv.QUOTE_NONNUMERIC)    # writes the dataset's rows
    print('  *| Clustering results for the CCL are exported to the file: %s\n' %ExportedFileName)
