# binary training result metrics
import pandas as pd
import numpy as np
from POI_project.support_code import general_func
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
# from sklearn.feature_selection import SelectFromModel
import matplotlib as plt
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
def preprocess2(df):
    pd.options.mode.use_inf_as_na = True
    df=df.replace([np.inf, -np.inf, "#NAME?"], 0)
    remove_cols=["sp#","win#","bldgI#"]
    preprocessed_df=df.drop(columns=remove_cols,axis=1)
    return preprocessed_df

def initial_balance(df, PIDs, POIs, min_data=200):
    df_balanced = pd.DataFrame(None, columns=df.columns)
    for PID in PIDs:
        tempdf = df[df["PID"] == PID]
        for POI in POIs:
            # df_balanced = df_balanced(df[df["PID"] == PID & df["POI#"] == POI]).head(min_data)
            df_balanced = df_balanced.append(tempdf[tempdf["POI#"] == POI].head(min_data))
    # print(df_balanced["PID"].value_counts())
    # print(df_balanced["POI#"].value_counts())
    return df_balanced


def balance_data(df, classes, PID, target_POI, min_data):
    # target_ID_data_count = (preprocess_na_df["PID"].values == target_ID).sum()
    #print("origioal df in balance data", df["PID"].value_counts())
    target_data_count = min_data
    #print("THis is target data count", target_data_count)
    df = df[df["PID"] == PID]
    df_balanced = df[df["POI#"] == target_POI].head(target_data_count)
    split = 0
    for POI in [3,4]:
        if(POI != target_POI):
            tempdf = df[df["POI#"] == POI]
            tempdf["POI#"] = tempdf["POI#"].replace({POI:-1})
            split = split + len(tempdf)
            df_balanced = df_balanced.append(tempdf)
    len_left = target_data_count - split
    for POI in [1,2]:
        if (POI != target_POI):
            tempdf = df[df["POI#"] == POI].head(len_left)
            tempdf["POI#"] = tempdf["POI#"].replace({POI: -1})
            df_balanced = df_balanced.append(tempdf)
    #print("From balancedata()")
    #print(df_balanced["POI#"].value_counts())
    return df_balanced


def feature_select_corr(rate, min_features, Xtrain_df, Xtest_df):
    Xtrain_df2 = Xtrain_df.drop(["majVoted_actiLvl", "is_workday", "day_of_week", "epoch_number", "hour"],
                                axis=1)  # for forced slection
    Xtest_df2= Xtest_df.drop(["majVoted_actiLvl", "is_workday", "day_of_week", "epoch_number", "hour"],
                                axis=1)  # for forced slection
    corr = Xtrain_df2.corr()  # look at only train data
    absCorr = np.abs(corr)

    one_minus_alpha = rate
    newXtrain_df = Xtrain_df2
    newXtest_df = Xtest_df2
    for row in range(len(absCorr)):
        if (len(newXtrain_df.columns) > min_features-5):
            for column in range(len(absCorr.iloc[row])):
                # print("row:", row, " column:", column)
                if (row == column):
                    continue
                elif (absCorr.iloc[row][column] >= one_minus_alpha):
                    # print(absCorr.iloc[row][column])
                    if (len(newXtrain_df.columns) > min_features - 5):
                        try:
                            newXtrain_df = newXtrain_df.drop(labels=absCorr.columns[column], axis=1)
                            newXtest_df = newXtest_df.drop(labels=absCorr.columns[column], axis=1)
                            #print("this is len ", len(newXtrain_df.columns))
                        except:
                            pass
    for item in ["majVoted_actiLvl", "is_workday", "day_of_week", "epoch_number", "hour"]:
        newXtrain_df[item] = Xtrain_df[item]
        newXtest_df[item] = Xtest_df[item]
        #print(newXtest_df.columns)
    return newXtrain_df, newXtest_df


def feature_select_PCA(features, Xtrain_df, Xtest_df):
    pca = PCA(n_components=features)
    pca.fit(Xtrain_df)
    newXtrain_df = pca.transform(Xtrain_df)
    newXtest_df = pca.transform(Xtest_df)
    return newXtrain_df, newXtest_df


def feature_select_KBest(features, Xtrain_df, Ytrain_df, Xtest_df):
    # fig = plt.figure()
    X_trainPro = SelectKBest(f_classif, k=features)
    Xtrain_df2 = Xtrain_df.drop(["majVoted_actiLvl", "is_workday", "day_of_week", "epoch_number", "hour"],
                                axis=1)  # for forced slection
    X_trainPro.fit(Xtrain_df2, Ytrain_df.values.ravel())
    kscores = X_trainPro.pvalues_  # -np.log10(X_trainPro.pvalues_)
    # print(kscores)
    '''
    ax1 = fig.add_subplot(1, 1, 1)
    ktemp = kscores.copy()
    kfeaturenames = np.array([])
    orderScore = np.array([])

    if (len(kscores) < 25):
        numDisplay = len(kscores)
    else:
        numDisplay = 25

    for i in range(numDisplay):
        kfeaturenames = np.append(kfeaturenames, Xtrain_df.columns[np.nonzero(ktemp == ktemp.max())[0][0]])
        orderScore = np.append(orderScore, ktemp.max())
        # print(np.nonzero(ktemp == ktemp.max()))
        ktemp[np.nonzero(ktemp == ktemp.max())[0][0]] = 0
    k11thscore = orderScore[features]
    kmask1 = orderScore <= k11thscore
    kmask2 = orderScore > k11thscore
    overbar = ax1.bar(kfeaturenames[kmask2], orderScore[kmask2], align="center", color="white", edgecolor="green")
    underbar = ax1.bar(kfeaturenames[kmask1], orderScore[kmask1], align="center", color="white", edgecolor="blue")
    ax1.set_title("SelectKBest")
    ax1.tick_params(labelrotation=90)

    # to print values
    iterValue = 0
    for bar in overbar:
        ax1.text(bar.get_x(), bar.get_height(), str(round(orderScore[iterValue], 1)))
        iterValue += 1

    for bar in underbar:
        ax1.text(bar.get_x(), bar.get_height(), str(round(orderScore[iterValue], 1)))
        iterValue += 1

    plt.subplots_adjust(hspace=2, bottom=.2, top=.85)
    # plt.show()
    plt.close()
    '''
    # selection of the top values to use
    Top_Rank = np.array(["majVoted_actiLvl"])
    # Top_Rank = np.array(["majVoted_actiLvl"]) #this is for force activities in it
    scores = kscores
    for i in range(features - 1):  # features-1 #this is for force activities in it
        Top_item = scores.max()
        Top_item_loc = np.where(scores == np.max(scores))[0][0]
        Top_Rank = np.append(Top_Rank, Xtrain_df.columns[Top_item_loc])
        scores[Top_item_loc] = 0
    X_train = Xtrain_df[Top_Rank]
    X_test = Xtest_df[Top_Rank]

    return X_train, X_test, Top_Rank


pid_selected = general_func.select_pid(row_threshold=200)
feature_count = 20
pid = 100  # check if i still need this
classes = [1, 2, 3, 4]  # check if i still need this
num_shuffles = 10
train_percent = .95
test_IDs = [26, 53, 66, 215, 239, 248, 360, 524]
target_POIs = [1, 2, 3, 4]
min_data = 200

# source_data_path="C:\\Users\\willi\\PycharmProjects\\Test\\POI_project\\subjects_data_0223" #Chih-You Data with general_func.preprocess
# source_data_path="C:\\Users\\willi\\PycharmProjects\\Test\\POI_project\\0603Data" #Will Data with general_func.preprocess
source_data_path = "C:\\Users\\willi\\PycharmProjects\\Test\\POI_project\\0603Data\\Clean\\DataCleaner_6_12_output"  # Will Data with general_func.preprocess2
target_data_path = "C:\\Users\\willi\\PycharmProjects\\Test\\POI_project\\Data\\6_09"

df = pd.read_csv(source_data_path + "\\bio_feat_time_4pois_pid26clean.csv")
for column in ["time", "index", "dayOfWeek", "isWorkDay", "snrS", "snrM", "snrC", "snrH"]:
    if (column in df.columns):
        df = df.drop([column], axis=1)
preprocess_na_df = preprocess2(df)
print("POI to train list", classes)
for pid in test_IDs:
    if(pid != 26):
        df = pd.read_csv(source_data_path + "\\bio_feat_time_4pois_pid" + str(pid) + "clean.csv")
        for column in ["time", "index", "dayOfWeek", "isWorkDay", "snrS", "snrM", "snrC", "snrH"]:
            if (column in df.columns):
                df = df.drop([column], axis=1)
        preprocess_na_df = preprocess_na_df.append(preprocess2(df),
                                                   ignore_index=True)  # .drop(['Unnamed: 0'],axis=1)
for column in df.columns:
    df[column] = df[column].fillna(0)
# print(preprocess_na_df["PID"].value_counts())
#df = initial_balance(preprocess_na_df, test_IDs, target_POIs, min_data)
df = preprocess_na_df
file_name = "Data_for_POI_Binary_POI_Individual_Model_7_4.xlsx"
dfFile = pd.ExcelWriter(file_name, engine='xlsxwriter')
df.to_excel(dfFile, sheet_name="origional")
Xdf = df.drop("POI#", axis=1)
# Xdf = df.drop(["is_workday", "day_of_week", "epoch_number", "hour"], axis=1) # This is for non temporial
Ydf = df["POI#"]
Xdf.to_excel(dfFile, sheet_name="xdf")
Ydf.to_excel(dfFile, sheet_name="ydf")
dfFile.save()
print("Saved Data file")

scores_df = pd.DataFrame(None, columns=["Target ID", "Shuffle #", "Model", "TP", "FN", "FP", "TN", "ROC-AUC",
                                        "best params", "selected features"])
for POI in [1,2]:
    target_POI = POI

    for test_ID in test_IDs:
        print("working on POI ", target_POI, " PID as test ", test_ID)
        min_data = min(len(df[(df["POI#"] == target_POI) & (df["PID"] == test_ID)]),
                       len(df[(df["POI#"] != target_POI) & (df["PID"] == test_ID)]))
        print("The Min of ", len(df[(df["POI#"] == target_POI) & (df["PID"] == test_ID)]), " and ",
              len(df[(df["POI#"] != target_POI) & (df["PID"] == test_ID)]), " is ",
              min_data)

        # print(len(df), ":", len(Train_df), len(Test_df)) # train test split check

        IDdf = balance_data(df, classes, test_ID, target_POI, min_data).drop("Unnamed: 0", axis=1)
        #print(IDdf["PID"].value_counts(), test_ID)
        #traindf = IDdf[IDdf["PID"] != test_ID]
        #testdf = IDdf[IDdf["PID"] == test_ID]
        # print(testdf)
        # print(traindf["POI#"].value_counts())
        # print(traindf["PID"].value_counts())
        X_train, X_test, Y_train, Y_test = train_test_split(IDdf.drop(["PID", "POI#"], axis=1), IDdf["POI#"].astype('int'), test_size=0.2, random_state=0)
        '''
        X_train = traindf.drop(["POI#", "PID"], axis=1)
        Y_train = traindf["POI#"]
        X_test = testdf.drop(["POI#", "PID"], axis=1)
        Y_test = testdf["POI#"]
        '''# This is used for Population Model
        # print(len(X_train), " should equal ", len(Y_train))
        l1_X_train, l1_X_test = feature_select_corr(.9, 25, X_train, X_test)
        print("finished feature selection layer 1")
        #print(l1_X_train.columns)
        # print(l1_X_train)
        #l2_X_train, l2_X_test = feature_select_PCA(feature_count, l1_X_train, l1_X_test)
        l2_X_train, l2_X_test, Top_Rank = feature_select_KBest(feature_count, l1_X_train, Y_train, l1_X_test)
        print("finished feature selection layer 2")
        print("Y_train.value_counts() \n", Y_train.value_counts())
        print("Y_test.value_counts() \n", Y_test.value_counts())

        sc = StandardScaler()
        X_train = sc.fit_transform(l2_X_train)
        X_test = sc.fit_transform(l2_X_test)

        # K nearest Neighbors
        neighbors = np.array(range(10, 20, 2))
        KNNparams = [{"n_neighbors": neighbors, "weights": ["distance"]}]
        KNN = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=KNNparams, cv=3)
        print("fitting KNN model...")
        KNN.fit(X_train, Y_train.values.ravel())
        pred_knn = KNN.predict(X_test)


        print("Knn GS\n", classification_report(Y_test.values.ravel(), pred_knn))
        print(confusion_matrix(Y_test.values.ravel(), pred_knn))
        print("KNN", feature_count, "TP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_knn)[0][0])
        print("KNN", feature_count, "FN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_knn)[0][1])
        print("KNN", feature_count, "FP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_knn)[1][0])
        print("KNN", feature_count, "TN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_knn)[1][1])
        print("KNN", feature_count, "roc_auc_score" + " " + str(POI) + " " + str(test_ID) + " ",
              roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_knn))
        print("KNN", feature_count, "features" + " " + str(POI) + " " + str(test_ID) + " ", Top_Rank)
        print("KNN", feature_count, "best_estimator" + " " + str(POI) + " " + str(test_ID) + " ", KNN.best_estimator_)
        # Example of print out version of scores

        matrix = confusion_matrix(Y_test.values.ravel(), pred_knn)
        TP, FN, FP, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        score_info = [target_POI, test_ID, "KNN", TP, FN, FP, TN,
                      roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_knn), KNN.best_estimator_, Top_Rank]
        scores_df = scores_df.append(pd.Series(score_info, index=scores_df.columns), ignore_index=True)

        # SVM RBF
        gamma = []
        c = []
        degree = []
        for i in range(10, 50, 10):
            gamma.append(i / 1000)
        for i in range(1, 3, 1):
            c.append(i)
        for i in range(2, 3):
            degree.append(i)

        SVMparameters = [{"kernel": ["rbf"], "C": c, "gamma": gamma, "random_state": [0]}]
        Modelsvm = GridSearchCV(estimator=SVC(probability=False), param_grid=SVMparameters, cv=3)
        print("fitting SVM(RBF) model...")
        Modelsvm.fit(X_train, Y_train.values.ravel())
        pred_svm = Modelsvm.predict(X_test)

        print("SVM(RBF) GS\n", classification_report(Y_test.values.ravel(), pred_svm))
        print(confusion_matrix(Y_test.values.ravel(), pred_svm))
        print("SVM(RBF)", feature_count, "TP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_svm)[0][0])
        print("SVM(RBF)", feature_count, "FN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_svm)[0][1])
        print("SVM(RBF)", feature_count, "FP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_svm)[1][0])
        print("SVM(RBF)", feature_count, "TN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_svm)[1][1])
        print("SVM(RBF)", feature_count, "roc_auc_score" + " " + str(POI) + " " + str(test_ID) + " ",
              roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_svm))
        print("SVM(RBF)", feature_count, "features" + " " + str(POI) + " " + str(test_ID) + " ", Top_Rank)
        print("SVM(RBF)", feature_count, "best_estimator" + " " + str(POI) + " " + str(test_ID) + " ", Modelsvm.best_estimator_)
        # Example of print out version of scores

        matrix = confusion_matrix(Y_test.values.ravel(), pred_svm)
        TP, FN, FP, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        score_info = [target_POI, test_ID, "SVM(RBF)", TP, FN, FP, TN,
                      roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_svm),
                      Modelsvm.best_estimator_, Top_Rank]
        scores_df = scores_df.append(pd.Series(score_info, index=scores_df.columns), ignore_index=True)

        gamma = [10 / 1000]
        # SVM Poly
        SVMPolyparameters = [{"kernel": ["poly"], "C": c, "gamma": gamma, "degree": degree, "random_state": [0]}]

        Modelsvm = GridSearchCV(estimator=SVC(probability=False), param_grid=SVMPolyparameters, cv=3)
        print("fitting SVM(POLY) model...")
        Modelsvm.fit(X_train, Y_train.values.ravel())
        pred_svm = Modelsvm.predict(X_test)

        print("SVM(Poly) GS\n", classification_report(Y_test.values.ravel(), pred_svm))
        print(confusion_matrix(Y_test.values.ravel(), pred_svm))
        print("SVM(Poly)", feature_count, "TP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_svm)[0][0])
        print("SVM(Poly)", feature_count, "FN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_svm)[0][1])
        print("SVM(Poly)", feature_count, "FP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_svm)[1][0])
        print("SVM(Poly)", feature_count, "TN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_svm)[1][1])
        print("SVM(Poly)", feature_count, "roc_auc_score" + " " + str(POI) + " " + str(test_ID) + " ",
              roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_svm))
        print("SVM(Poly)", feature_count, "features" + " " + str(POI) + " " + str(test_ID) + " ", Top_Rank)
        print("SVM(Poly)", feature_count, "best_estimator" + " " + str(POI) + " " + str(test_ID) + " ",
              Modelsvm.best_estimator_)
        # Example of print out version of scores

        matrix = confusion_matrix(Y_test.values.ravel(), pred_svm)
        TP, FN, FP, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        score_info = [target_POI, test_ID, "SVM(POLY)", TP, FN, FP, TN,
                      roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_svm),
                      Modelsvm.best_estimator_, Top_Rank]
        scores_df = scores_df.append(pd.Series(score_info, index=scores_df.columns), ignore_index=True)

        # Naive Bayes
        var = [i / 1000000000000 for i in range(5, 11, 1)]
        param = [{"var_smoothing": var}]

        Gaus = GridSearchCV(estimator=GaussianNB(), param_grid=param, cv=3)
        print("fitting NB model...")
        Gaus.fit(X_train, Y_train.values.ravel())
        pred_nb = Gaus.predict(X_test)

        print("NB GS\n", classification_report(Y_test.values.ravel(), pred_nb))
        print(confusion_matrix(Y_test.values.ravel(), pred_nb))
        print("NB", feature_count, "TP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_nb)[0][0])
        print("NB", feature_count, "FN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_nb)[0][1])
        print("NB", feature_count, "FP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_nb)[1][0])
        print("NB", feature_count, "TN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_nb)[1][1])
        print("NB", feature_count, "roc_auc_score" + " " + str(POI) + " " + str(test_ID) + " ",
              roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_nb))
        print("NB", feature_count, "features" + " " + str(POI) + " " + str(test_ID) + " ", Top_Rank)
        print("NB", feature_count, "best_estimator" + " " + str(POI) + " " + str(test_ID) + " ",
              Gaus.best_estimator_)

        matrix = confusion_matrix(Y_test.values.ravel(), pred_nb)
        TP, FN, FP, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        score_info = [target_POI, test_ID, "NB", TP, FN, FP, TN,
                      roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_nb),
                      Gaus.best_estimator_, Top_Rank] #only used for KBest
        scores_df = scores_df.append(pd.Series(score_info, index=scores_df.columns), ignore_index=True)

        # Random Forest
        param = [{"n_estimators": [50, 100, 150, 200], "min_samples_leaf": [10, 20, 30]}]
        # "criterion": ["gini", "entropy"], "min_samples_split": [3, 4, 5], "min_samples_leaf": [16, 20, 24]}]

        BestForest = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param, cv=3)
        print("fitting RF model...")
        BestForest.fit(X_train, Y_train.values.ravel())
        pred_rf = BestForest.predict(X_test)

        print("RF GS\n", classification_report(Y_test.values.ravel(), pred_rf))
        print(confusion_matrix(Y_test.values.ravel(), pred_rf))
        print("RF", feature_count, "TP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_rf)[0][0])
        print("RF", feature_count, "FN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_rf)[0][1])
        print("RF", feature_count, "FP:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_rf)[1][0])
        print("RF", feature_count, "TN:" + " " + str(POI) + " " + str(test_ID) + " ",
              confusion_matrix(Y_test.values.ravel(), pred_rf)[1][1])
        print("RF", feature_count, "roc_auc_score" + " " + str(POI) + " " + str(test_ID) + " ",
              roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_rf))
        print("RF", feature_count, "features" + " " + str(POI) + " " + str(test_ID) + " ", Top_Rank)
        print("RF", feature_count, "best_estimator" + " " + str(POI) + " " + str(test_ID) + " ",
              BestForest.best_estimator_)

        matrix = confusion_matrix(Y_test.values.ravel(), pred_rf)
        TP, FN, FP, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        score_info = [target_POI, test_ID, "RF", TP, FN, FP, TN,
                      roc_auc_score(y_true=Y_test.values.ravel(), y_score=pred_rf),
                      BestForest.best_estimator_, Top_Rank]
        scores_df = scores_df.append(pd.Series(score_info, index=scores_df.columns), ignore_index=True)

    print("Data saving")
    writer = pd.ExcelWriter("POI_Binary_Personal_Model_1&2_extended_POI" + str(POI) + "_Results_7_4.xlsx", engine='xlsxwriter')
    scores_df.to_excel(writer, sheet_name='Sheet1')
    #writer.save()
    print(scores_df)