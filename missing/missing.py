"""
@ author : sourav kumar (101883068)
@ made for UCS633 - PROJECT - III
@ Timestamp : 2020 / 2 / 14
"""
# Importing all packages required for imputation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.imputation import mice
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import learning_curve
import sys
import re
# Class for handling missing values algorithm module
class missing:
    """
    Attributes:
    df : original dataframe of input file
    input_file : file to be read and perform missing values removal algorithm
    output_file : file to be written into after performing the algorithm
    """
    def __init__(self, read_file, write_file):
        """
        Args:
        input_file : file to be read and perform missing values removal algorithm
        output_file : file to be written into after performing the algorithm
        """
        # check for proper csv file
        assert "csv" in f"{read_file}", "Could not recognize csv file, try checking your input file"
        assert "csv" in f"{write_file}", "Could not recognize csv file, try checking your output file"
        # load and read csv file
        self.df = pd.read_csv(read_file).iloc[:, 1:]
        print('File read succesfully !', f'Shape of original file : {self.df.shape}')
        self.input_file = read_file
        self.output_file = write_file


    # function for detecting missing values and reporting it
    def detect_missing(self):
        # checking missing values
        null_series = self.df.isnull().sum()
        print()
        null_column_list = []
        if sum(null_series):
            print('Following columns contains missing values : ')
            total_samples = self.df.shape[0]
            for i, j in null_series.items():
                if j:
                    print("{} : {:.2f} %".format(i, (j/total_samples)*100))
                    null_column_list.append(i)
        else:
            print("None of the columns contains missing values !")
        return null_column_list

    # using row removal
    def row_removal(self):
        original_row, original_col = self.df.shape[0], self.df.shape[1]
        print()
        print('Using row removal algorithm...')
        # removing rows
        df_row = self.df.dropna(axis=0)
        print(f"Shape of new dataframe : {df_row.shape}")
        print(f"Total {original_row - df_row.shape[0]} rows removed")
        return df_row

    # using column removal
    def column_removal(self):
        original_row, original_col = self.df.shape[0], self.df.shape[1]
        print()
        print('Using column removal algorithm...')
        print('Warning : Features may be reduced, introducing inconsistency when Testing !')
        # removing columns
        df_col = self.df.dropna(axis=1)
        print(f"Shape of new dataframe : {df_col.shape}")
        print(f"Total {original_col - df_col.shape[1]} columns removed")
        return df_col

    # using statistical imputation
    def stats_imputation(self, null_column_list):
        print()
        print('Using Statistical imputation algorithm...')
        # extracting columns for numerical columns
        valid_cols = [column for column in null_column_list if self.df[column].dtype != 'object']
        # extracting columns for categorical columns
        categorical_cols = [column for column in null_column_list if self.df[column].dtype == 'object']
        numeric_cols = valid_cols
        df_stats_mean, df_stats_median, df_stats_mode = self.df.copy(), self.df.copy(), self.df.copy()
        # Imputing mean for numeric values and then imputing median and mode for categorical values
        print(f'Imputing following columns with mean, median and mode : {numeric_cols}')
        print(f'Imputing following columns with mode : {categorical_cols}')
        if len(numeric_cols):
            for i in numeric_cols:
                df_stats_mean.fillna({i : self.df[i].mean()}, inplace=True)
                df_stats_median.fillna({i : self.df[i].median()}, inplace=True)
                df_stats_mode.fillna({i : self.df[i].mode()[0]}, inplace=True)

        if len(categorical_cols):
            for i in categorical_cols:
                df_stats_mean.fillna({i : self.df[i].mode()[0]}, inplace=True)
                df_stats_median.fillna({i : self.df[i].mode()[0]}, inplace=True)
                df_stats_mode.fillna({i : self.df[i].mode()[0]}, inplace=True)

        return df_stats_mean, df_stats_median, df_stats_mode

    # using interpolation algorithm
    def interpolate_impute(self):
        print()
        print('Using interpolation algorithm using linear method...')
        df_interpolate = self.df.copy()
        # mapping embarked values by numeric values
        embarked_mapping = {"S": 1, "C": 2, "Q": 3}
        df_interpolate['Embarked'] = df_interpolate['Embarked'].map(embarked_mapping)
        # mapping Cabin string values by numeric values
        deck = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "U": 7}
        df_interpolate['Cabin'] = df_interpolate['Cabin'].fillna("U")
        df_interpolate['Cabin'] = df_interpolate['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        df_interpolate['Cabin'] = df_interpolate['Cabin'].map(deck)
        df_interpolate['Cabin'].replace({7:np.nan}, inplace=True)
        df_interpolate.interpolate(method='linear', inplace=True, limit_direction='both')
        # reverse mapping the values
        embarked_mapping = {1:"S", 2:"C", 3:"Q"}
        df_interpolate['Embarked'] = df_interpolate['Embarked'].map(embarked_mapping)
        deck_mapping = {0 : "A", 1 : "B", 2 : "C", 3 : "D", 4 : "E", 5 : "F", 6 : "G"}
        df_interpolate['Cabin'] = df_interpolate['Cabin'].map(deck_mapping)
        return df_interpolate

    # using MICE algorithm
    def MICE_impute(self):
        print()
        print('Using MICE algorithm...')
        df_mice = self.df.copy()
        # mapping Embarked using numeric values
        embarked_mapping = {"S": 1, "C": 2, "Q": 3}
        df_mice['Embarked'] = df_mice['Embarked'].map(embarked_mapping)
        # mapping Cabin using numeric values
        deck = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "U": 7}
        df_mice['Cabin'] = df_mice['Cabin'].fillna("U")
        df_mice['Cabin'] = df_mice['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        df_mice['Cabin'] = df_mice['Cabin'].map(deck)
        df_mice['Cabin'].replace({7:np.nan}, inplace=True)

        numeric_features = [column for column in df_mice.columns if df_mice[column].dtype != 'object']
        imp = mice.MICEData(df_mice[numeric_features])
        imp.set_imputer('')
        for i in range(100):
            imp.update_all()
        operated_cols = [column for column in numeric_features if self.df[column].isnull().sum()]
        print(f'Operating on following features : {operated_cols}')
        # copying the imputed values to the original df
        for i in operated_cols:
            df_mice[i] = imp.data[i]

        # reverse mapping the values
        embarked_mapping = {1:"S", 2:"C", 3:"Q"}
        df_mice['Embarked'] = df_mice['Embarked'].map(embarked_mapping)
        deck_mapping = {0 : "A", 1 : "B", 2 : "C", 3 : "D", 4 : "E", 5 : "F", 6 : "G"}
        df_mice['Cabin'] = df_mice['Cabin'].map(deck_mapping)
        return df_mice

    # preprocessing the dataset
    def preprocess(self, df_train):
        print()
        # dropping the features
        fields_to_drop = ['Name', 'Ticket']
        df_train = df_train.drop(fields_to_drop, axis=1)
        dummy_fields = [column for column in df_train.columns if df_train[column].dtype == 'object']
        # concatenating the categorical features
        for each in dummy_fields:
            dummies = pd.get_dummies(df_train[each], drop_first=False)
            df_train = pd.concat([df_train, dummies], axis=1)
            df_train = df_train.drop([each], axis=1)

        return df_train

    # splitting the dataset
    def split_dataset(self, df_train):
        # X is training features, y is labels
        X = df_train.iloc[:, 1:]
        y = df_train.iloc[:, 0]
        return train_test_split(X, y, test_size = 0.3, random_state = 0)

    # function to predict the values after fitting the model
    # Here, i have chosen to use LogisticRegression model as a baseline
    # so that , effect of different algorithms can be seen clearly.
    def predict(self, x_train, x_test, y_train, y_test):
        logreg = LogisticRegression(solver='lbfgs', max_iter=700, random_state=0)
        logreg.fit(x_train, y_train)
        logreg_predictions = logreg.predict(x_test)
        return logreg_predictions

    # function to evaluate model and print the metrics
    def evaluate(self, y_pred, y_test):
        print(f"Accuracy : {round(accuracy_score(y_test, y_pred) * 100, 2)}")
        print(f"Log_loss : {log_loss(y_test, y_pred)}")
        precision, recall = precision_recall_fscore_support(y_test, y_pred)[0], precision_recall_fscore_support(y_test, y_pred)[1]
        print(f"precision : {precision} , recall : {recall}")

    # plot cross validation scores for all the algorithms
    def plot_metrics(self, df_list):
        # df_list is a tuple of all the x_train and x_test
        test_scores_mean = []
        for df in df_list:
            estimator = LogisticRegression(solver='lbfgs', max_iter=700, random_state=0)
            train_sizes, train_scores, test_scores = learning_curve(estimator, df[0], df[1], cv=5, random_state=0)
            test_scores_mean.append(np.mean(test_scores, axis=1))

        print("Plotting final metrics cross validation scores for all algorithms : ")
        plt.xlabel("Training examples")
        plt.ylabel("Cross-validation score")
        plt.title('LOGISTIC REGRESSION ALGORITHM')
        plt.plot(train_sizes, test_scores_mean[0], 'o-', color="b", label="row removal")
        plt.plot(train_sizes, test_scores_mean[1], 'o-', color="g", label="column removal")
        plt.plot(train_sizes, test_scores_mean[2], 'o-', color="k", label="mean imputed")
        plt.plot(train_sizes, test_scores_mean[3], 'o-', color="c", label="median imputed")
        plt.plot(train_sizes, test_scores_mean[4], 'o-', color="m", label="mode imputed")
        plt.plot(train_sizes, test_scores_mean[5], 'o-', color="y", label="interpolation imputed")
        plt.plot(train_sizes, test_scores_mean[6], 'o-', color="r", label="MICE imputed")
        plt.legend(loc='best')
        plt.show()

    # main executing functions
    def missing_main(self):
        """
        Args:
        method : Which algorithm to choose from currently following are the implemented ones \

        """
        # printing missing detected values
        null_column_list = self.detect_missing()
        # applying row removal
        df_row = self.row_removal()
        # applying column removal
        df_col = self.column_removal()
        # applying statistical imputation
        df_stats_mean, df_stats_median, df_stats_mode = self.stats_imputation(null_column_list)
        # applying interpo imputation
        df_interpolate = self.interpolate_impute()
        # applying MICE computed imputation
        df_mice = self.MICE_impute()

        # list to be stored as tupled train, test features for plotting cross validation scores
        features_training_list = []
        # metrics modelling for row removal algorithm
        print()
        print("Metrics for row removal algorithm")
        df_train = self.preprocess(df_row)
        x_train, x_test, y_train, y_test = self.split_dataset(df_train)
        y_pred = self.predict(x_train, x_test, y_train, y_test)
        self.evaluate(y_pred, y_test)
        features_training_list.append((x_train, y_train))
        # metrics column removal
        print()
        print("Metrics for column removal algorithm")
        df_train = self.preprocess(df_col)
        x_train, x_test, y_train, y_test = self.split_dataset(df_train)
        y_pred = self.predict(x_train, x_test, y_train, y_test)
        self.evaluate(y_pred, y_test)
        features_training_list.append((x_train, y_train))
        # metrics mean imputation
        print()
        print("Metrics for mean imputation algorithm")
        df_train = self.preprocess(df_stats_mean)
        x_train, x_test, y_train, y_test = self.split_dataset(df_train)
        y_pred = self.predict(x_train, x_test, y_train, y_test)
        self.evaluate(y_pred, y_test)
        features_training_list.append((x_train, y_train))
        # metrics median imputation
        print()
        print("Metrics for median imputation algorithm")
        df_train = self.preprocess(df_stats_median)
        x_train, x_test, y_train, y_test = self.split_dataset(df_train)
        y_pred = self.predict(x_train, x_test, y_train, y_test)
        self.evaluate(y_pred, y_test)
        features_training_list.append((x_train, y_train))
        # metrics mode imputation
        print()
        print("Metrics for mode imputation algorithm")
        df_train = self.preprocess(df_stats_mode)
        x_train, x_test, y_train, y_test = self.split_dataset(df_train)
        y_pred = self.predict(x_train, x_test, y_train, y_test)
        self.evaluate(y_pred, y_test)
        features_training_list.append((x_train, y_train))
        # metrics interpolatation algorithm
        print()
        print("Metrics for interpolation imputation algorithm")
        df_train = self.preprocess(df_interpolate)
        x_train, x_test, y_train, y_test = self.split_dataset(df_train)
        y_pred = self.predict(x_train, x_test, y_train, y_test)
        self.evaluate(y_pred, y_test)
        features_training_list.append((x_train, y_train))
        # metrics MICE algorithm
        print()
        print("Metrics for MICE imputation algorithm")
        df_train = self.preprocess(df_mice)
        x_train, x_test, y_train, y_test = self.split_dataset(df_train)
        y_pred = self.predict(x_train, x_test, y_train, y_test)
        self.evaluate(y_pred, y_test)
        features_training_list.append((x_train, y_train))

        print("Maximum Accuracy and minimum loss is obtained from using MICE imputation algorithm !")
        self.plot_metrics(features_training_list)

        # saving the file and writing to output path provided
        df_mice.to_csv(self.output_file, index = False)
        print()
        print(f"DataFrame of shape {df_mice.shape} written to {self.output_file}")

        # useful when , you need to see the actual Statistics of missing values
        # Plotting results for best missing values handling algorithm
        # # plotting results
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(30,30))
        plt.subplots_adjust(wspace=0.25, hspace=0.4)
        sns.heatmap(self.df.isnull(), yticklabels=False, cbar=False, cmap='rocket', ax = axes[0][0]).set_title("Original")
        sns.heatmap(df_row.isnull(), yticklabels=False, cbar=False, cmap='rocket', ax = axes[0][1]).set_title("Row removal")
        sns.heatmap(df_col.isnull(), yticklabels=False, cbar=False, cmap='rocket', ax = axes[0][2]).set_title("Column_removal")
        sns.heatmap(df_stats_mean.isnull(), yticklabels=False, cbar=False, cmap='rocket', ax = axes[0][3]).set_title("Mean imputation")
        sns.heatmap(df_stats_median.isnull(), yticklabels=False, cbar=False, cmap='rocket', ax = axes[1][0]).set_title("Median imputation")
        sns.heatmap(df_stats_mode.isnull(), yticklabels=False, cbar=False, cmap='rocket', ax = axes[1][1]).set_title("Mode imputation")
        sns.heatmap(df_interpolate.isnull(), yticklabels=False, cbar=False, cmap='rocket', ax = axes[1][2]).set_title("interpolation imputation")
        sns.heatmap(df_mice.isnull(), yticklabels=False, cbar=False, cmap='rocket', ax = axes[1][3]).set_title("MICE imputation")
        plt.show()


# main driver function
if __name__ == '__main__':
    print('WELCOME TO HANDLING MISSING VALUES ALGORITHM')
    print('EXPECTED ARGUMENTS TO BE IN ORDER : python -m missing.missing <InputFile.csv> <OutputFile.csv>')
    # runs if and only if it contains atleast inputfile, output file
    if len(sys.argv) == 3:
        read_file = sys.argv[1]
        write_file = sys.argv[2]
        print(f"file given : {read_file}")
        m = missing(read_file, write_file)
        m.missing_main()
    # report for incorrect order of arguments passed
    else:
        print('PLEASE PASS ARGUMENTS IN ORDER : python -m missing.missing <InputFile.csv> <OutputFile.csv>')
