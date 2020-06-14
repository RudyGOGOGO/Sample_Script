# -*-coding:utf-8-*-
# Learning about the variables:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM


def drop_duplicate(data, key_index,
                   in_place=True, keep_pos="First"):
    if training.shape[0] == len(set(data.iloc[:, key_index])):
        print("There is no duplicate records")
    else:
        training.drop_duplicates(keep=keep_pos, inplace=in_place)


def fill_nan_4_true_missing(data):
    for row in range(data.shape[0]):
        if 1 not in list(data.iloc[row, :]):
            data.iloc[row, :] = None
    return data


def fill_alternative_value(data, column, alter_label):
    null_list = data.loc[:, column].isnull()
    for row in range(len(null_list)):
        if null_list[row]:
            data.loc[row, column] = alter_label


def gene_dummy_variables(data, nominal_var_list):
    for var in nominal_var_list:
        with_special_na = var[1]
        if with_special_na:
            fill_alternative_value(data, var[0], "Unavailable")
        else:
            fill_alternative_value(data, var[0], None)
        dummies = pd.get_dummies(data.loc[:, var[0]],
                                 prefix=var[0])
        fill_nan_4_true_missing(dummies)
        data = pd.concat([data, dummies], axis=1)
    drop_list = [nominal_var_list[i][0] for i in range(len(nominal_var_list))]
    data.drop(drop_list, axis=1, inplace=True)
    return data


def convert_ordinal_variables(data, ordinal_var_list):
    for var in ordinal_var_list:
        with_special_na = var[1]
        if with_special_na:
            fill_alternative_value(data, var[0], 0)
        else:
            fill_alternative_value(data, var[0], None)
        print(str(data.loc[:, var[0]].dtype).upper())
        if str(data.loc[:, var[0]].dtype).upper() == "OBJECT":
            for row in range(data.shape[0]):
                data.loc[:, var[0]].apply(lambda x:
                                          var[1].index(x.upper()) if x.upper() in var[1] else None)
                print(data.loc[:, var[0]])


def detect_missing(data):
    columns = data.columns
    no_missing = True
    for i in range(data.shape[1]):
        if data.iloc[:, i].isnull().values.any():
            no_missing = False
            print("The column " + columns[i] +
                  " contains missing values: ")
    if no_missing:
        print("No column contains missing values")


def do_logarithm(data):
    columns = data.columns
    for i in range(data.shape[1]):
        column_type = str(data.iloc[:, i].dtypes)
        is_number_type = 'int' in column_type or 'float' in column_type
        if is_number_type:
            range_order = (data.iloc[:, i].max() -
                           data.iloc[:, i].min()) / 10
            if range_order > 1:
                for j in range(data.shape[0]):
                    if data.iloc[j, i] < 0:
                        print("The column " + columns[i] + " contains negative values")
                        print("Therefore, the log transformation could not be implemented")
                        break
                print("According to the log rule, the column "
                      + columns[i] + " is transformed by logarithm")
                data[columns[i]] = np.log(data.iloc[:, i] + 1)


def detect_outlier(data, classifier="Robust Covariance",
                   outlier_fraction=0.005):
    classifiers = {
        "Empirical Covariance": EllipticEnvelope(support_fraction=1.,
                                                 contamination=outlier_fraction),
        "Robust Covariance":
            EllipticEnvelope(contamination=outlier_fraction),
        "OCSVM": OneClassSVM(nu=outlier_fraction, gamma=0.05)}
    # colors = ['m', 'g', 'b']
    legend = {}
    # Learn a frontier for outlier detection with several classifiers
    xx1, yy1 = np.meshgrid(np.linspace(5, 10, 500),
                           np.linspace(10, 15, 500))
    plt.figure(1)
    clf = classifiers[classifier]
    clf.fit(data)
    scores = clf.decision_function(np.c_[xx1.ravel(),
                                         yy1.ravel()]).reshape(xx1.shape)
    legend[classifier] = plt.contour(
        xx1, yy1, scores,
        levels=[0], linewidths=2, colors='m',
        linestyles='dashed')
    legend_key = list(legend.keys())
    # Plot the results (= shape of the data points cloud)
    plt.figure(1)  # two clusters
    plt.title("Identify potential outliers")
    plt.xlabel('log: Above grade (ground) living area square feet')
    plt.ylabel('log: Sales Price')
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color='black')
    plt.xlim((xx1.min(), xx1.max()))
    plt.ylim((yy1.min(), yy1.max()))
    plt.legend([legend_key[0]]).legendHandles[0].set_color('m')
    plt.show()


if __name__ == "__main__":
    # ================Step1:Data Pre-processing==========
    # 1. load the training and testing data set into the data-frame object
    root_path = 'G:/Learning/(Going)DS_Interview/Projects/House Price Prediction/Data/'
    root_path2 = '/Users/weizhang/Desktop/'
    training = pd.read_csv(root_path2 + 'train.csv', keep_default_na=True)
    testing = pd.read_csv(root_path2 + 'test.csv', keep_default_na=True)
    column_list = list(training.columns)
    # 2. separate variable by their types
    # Type: 34 Numeric(Continuous and Discrete) + 23 Ordinal + 23 Nominal
    nominal_variables = [["MSSubClass", False], ["MSZoning", False],
                         ["Street", False], ["Alley", True],
                         ["Utilities", False],
                         ["LotConfig", False], ["Neighborhood", False],
                         ["Condition1", False], ["Condition2", False],
                         ["BldgType", False], ["HouseStyle", False],
                         ["RoofStyle", False], ["RoofMatl", False],
                         ["Exterior1st", False], ["Exterior2nd", False],
                         ["Foundation", False],
                         ["MasVnrType", True],
                         ["Heating", False], ["CentralAir", False],
                         ["Electrical", False], ["GarageType", True],
                         ["MiscFeature", True], ["SaleType", False],
                         ["SaleCondition", False]]
    # Note: Treat the year and month as the numeric type
    numeric_variables = ["LotFrontage", "LotArea", "YearBuilt",
                         "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                         "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
                         "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "Bedroom",
                         "Kitchen", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars",
                         "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
                         "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold", "SalePrice"]

    ordinal_variables = [["LotShape", False, ["REG", "IR1", "IR2", "IR3"]],
                         ["LandSlope", False, ["GTL", "MOD", "SEV"]],
                         ["LandContour", False, ["LVL", "BNK", "HLS", "LOW"]],
                         ["OverallQual", False],
                         ["OverallCond", False],
                         ["ExterQual", False, ["EX", "GD", "TA", "FA", "PO"]],
                         ["ExterCond", False, ["EX", "GD", "TA", "FA", "PO"]],
                         ["BsmtQual", True, ["EX", "GD", "TA", "FA", "PO"]],
                         ["BsmtCond", True, ["EX", "GD", "TA", "FA", "PO"]],
                         ["BsmtExposure", True, ["GD", "AV", "MN"]],
                         ["BsmtFinType1", True, ["GLQ", "ALQ", "BLQ", "REC", "LWQ", "UNF"]],
                         ["BsmtFinType2", True, ["GLQ", "ALQ", "BLQ", "REC", "LWQ", "UNF"]],
                         ["HeatingQC", False, ["EX", "GD", "TA", "FA", "PO"]],
                         ["KitchenQual", False, ["EX", "GD", "TA", "FA", "PO"]],
                         ["Functional", False, ["TYP", "MIN1", "MIN2", "MOD", "MAJ1", "MAJ2", "SEV", "SAL"]],
                         ["FireplaceQu", True, ["EX", "GD", "TA", "FA", "PO"]],
                         ["GarageQual", True, ["EX", "GD", "TA", "FA", "PO"]],
                         ["GarageCond", True, ["EX", "GD", "TA", "FA", "PO"]],
                         ["PoolQC", True, ["EX", "GD", "TA", "FA"]],
                         ["Fence", True, ["GDPRV", "MNPRV", "GDWO", "MNWW"]],
                         ["GarageFinish", True, ["FIN", "RFN", "UNF"]],
                         ["PavedDrive", False, ["Y", "P", "N"]]]
    # 3. check and drop duplicates of the training data set
    drop_duplicate(training, key_index=0)

    # 4. convert the nominal variables into dummy variables
    training = gene_dummy_variables(training, nominal_variables)
    testing = gene_dummy_variables(testing, nominal_variables)
    training = convert_ordinal_variables(training, ordinal_variables)

    # 5. convert the ordinal variables' values into discrete numbers




    # 5. check the missing value for continuous variables
    # detect_missing(training)

    # 6. transform the variable based on the log rule
    # do_logarithm(training)
    # training.to_csv(root_path+"trial.csv", sep=',', encoding='utf-8')

    # 7. drop unnecessary columns
    # training.drop(columns='Id', inplace=True)
    # testing.drop(columns='Id', inplace=True)

    # 8. identify the outliers
    # plt.scatter(training.GrLivArea, training.SalePrice,
    #             c='black', marker='s', cmap=plt.get_cmap('Spectral'))
    # plt.show()
    # outliers_labels = [31, 969, 1299]
    # detect_outlier(training.loc[:, ["GrLivArea", "SalePrice"]])

    # Based on the data description from the data set provider
    # there is no known
    # training.to_csv(root_path + "training_dummies.csv")
