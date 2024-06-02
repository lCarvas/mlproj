import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
import seaborn as sns

sns.set_theme()


class Preprocessing:
    @staticmethod
    def fillNa(data: pd.DataFrame, metricFeatures: list[str], boolFeatures: list[str]) -> pd.DataFrame:
        """Fill missing values

        Args:
            data (`pd.DataFrame`): Dataframe to be treated

        Returns:
            `pd.DataFrame`: Treated dataframe
        """    

        # on all of these features, if a value were to be different than 0, then it would not be missing, eg units approved, if the student approved, the value wouldn't be missing
        ifNaThen0: tuple[str,...] = (
            "N units credited 1st period",
            "N unscored units 1st period",
            "N scored units 1st period",
            "N units credited 2nd period",
            "N unscored units 2nd period",
            "N scored units 2nd period"
        )

        # these features are filled differently, basically incoherence checking, but filling the Na on 'N units approved 1st/2nd period' is needed beforehand, more info below
        checkAfterVars: list[list[str]] = [
            ["N units taken 1st period", "N scored units 1st period"],
            ["N units taken 2nd period", "N scored units 2nd period"]
        ]

        for var in metricFeatures:
            if var == (checkAfterVars[0][0] or checkAfterVars[1][0]): 
                continue # skip current iteration
            if var in ifNaThen0:
                data[var] = data[var].fillna(0) # fill the ifNaThen0 vars with well, 0s
            else:    
                data[var] = data[var].fillna(data[var].median()) # fill everything else with the median of the values of the feature

        # here we use the n units taken features we skipped earlier, a student has to have taken at least the same number of courses as the number of courses they passed
        for varList in checkAfterVars:
            # search for Na values on N units taken and replace by the equivalent value on N units approved
            data.loc[data[varList[0]].isna(), varList[0]] = data[varList[1]]
            # search for values on N units taken that are smaller than the equivalent on N units approved, replace by the equivalent value on N units approved
            data.loc[data[varList[0]] < data[varList[1]], varList[0]] = data[varList[1]]

        for var in boolFeatures:
            if var == "Regularized Fees":
                data[var] = data[var].fillna(1) # if nothing is said about the fees, we can assume they have been paid
            else:
                data[var] = data[var].fillna(0) # here is like the ifNaThen0 situation, if the values were to not be 0, they would have been declared

        return data
    
    @staticmethod
    def removeOutliers(data: pd.DataFrame) -> pd.DataFrame:
        """Removes outliers and fixes any negative number incoherences on the selected variables from the dataframe

        Args:
            data (`pd.DataFrame`): Dataframe to be treated

        Returns:
            `pd.DataFrame`: Treated dataframe
        """    

        toBeTreated: dict[str, dict[str, float | None]] = {
            "Application order": {"lower": 0, "upper": None},
            "Previous qualification score": {"lower": 0, "upper": None},
            "Entry score": {"lower": 0, "upper": None},
            "Age at enrollment": {"lower": 0, "upper": None},
            "N units credited 1st period": {"lower": 0, "upper": 15},
            "N units taken 1st period": {"lower": 0, "upper": 20},
            "N scored units 1st period": {"lower": 0, "upper": 25},
            "N units approved 1st period": {"lower": 0, "upper": 20},
            "Average grade 1st period": {"lower": 0, "upper": None},
            "N unscored units 1st period": {"lower": 0, "upper": None},
            "N units credited 2nd period": {"lower": 0, "upper": 14},
            "N units taken 2nd period": {"lower": 0, "upper": 15},
            "N scored units 2nd period": {"lower": 0, "upper": 25},
            "N units approved 2nd period": {"lower": 0, "upper": 15},
            "Average grade 2nd period": {"lower": 0, "upper": None},
            "N unscored units 2nd period": {"lower": 0, "upper": None},
            "Social Popularity": {"lower": 0, "upper": 100},
        }
        
        for var in toBeTreated:
            if toBeTreated[var]["lower"] != None:
                toRemove: list = list(data.loc[data[var] < toBeTreated[var]["lower"], var].index)
            if toBeTreated[var]["upper"] != None:
                toRemove.extend(list(data.loc[data[var] > toBeTreated[var]["upper"], var].index))
            data.drop(toRemove, axis=0, inplace=True)

        return data
    
    @staticmethod
    def groupValues(data: pd.DataFrame, grouping: Literal["low", "high"] = "high") -> pd.DataFrame:
        """replace values on columns that have lots of different values that can be grouped together to reduce the total number of dummies created after

        Args:
            data (`pd.DataFrame`): Dataframe to be treated
            grouping (Literal['low', 'high']): _description_

        Returns:
            `pd.DataFrame`: Treated dataframe
        """    
        
        if grouping == "low": 
            for col in ["Mother's qualification",  "Father's qualification", "Previous qualification"]:
                data.replace(regex={col: {r"(?i)^no school.*$": '0',
                                    r"(?i)^[0-4][a-z]{2} grade.*$": '1', 
                                    r"(?i)^[5-9]th grade.*$": '2', 
                                    r"(?i)^1[0-2]th grade.*$": '3', 
                                    r"(?i)^incomplete bachelor.*$": '4', 
                                    r"(?i)^bachelor degree.*$": '5',
                                    r"(?i)^post-grad.*$": '6',
                                    r"(?i)^master degree.*$": '7',
                                    r"(?i)^phd.*$": '8',}}, inplace=True)
        elif grouping == "high":
            for col in ["Mother's qualification",  "Father's qualification", "Previous qualification"]:
                data.replace(regex={col: {r"(?i)^no school.*$": 'None',
                                    r"(?i)^[0-4][a-z]{2} grade.*$": 'Lower', 
                                    r"(?i)^[5-9]th grade.*$": 'Lower', 
                                    r"(?i)^1[0-2]th grade.*$": 'Middle', 
                                    r"(?i)^incomplete bachelor.*$": 'Higher', 
                                    r"(?i)^bachelor degree.*$": 'Higher',
                                    r"(?i)^post-grad.*$": 'Higher',
                                    r"(?i)^master degree.*$": 'Higher',
                                    r"(?i)^phd.*$": 'Higher',}}, inplace=True)
        
        for col in ["Mother's occupation", "Father's occupation"]:
            data.replace(to_replace={col: ["Superior-level Professional", "Intermediate-level Professional", "Politician/CEO", "Teacher", "Information Technology Specialist"]}, value="Professional Fields", inplace=True)
            data.replace(to_replace={col: ["Skilled construction workers", "Assembly Worker", "Factory worker", "Lab Technocian"]}, value="Technical and Skilled Trades", inplace=True)
            data.replace(to_replace={col: ["Administrative Staff", "Office worker", "Accounting operator"]}, value="White collar Jobs", inplace=True)
            data.replace(to_replace={col: ["Restaurant worker", "Personal care worker", "Seller", "Cleaning worker"]}, value="Service Industry", inplace=True)
            data.replace(to_replace={col: ["Private Security", "Armed Forces"]}, value="Security and Armed Forces", inplace=True)
            data.replace(to_replace={col: ["Unskilled Worker", "Other", "Student", "Artist"]}, value="Recreational or unskilled", inplace=True)
            data.replace(to_replace={col: ["Engineer", "Scientist", "Health professional"]}, value="STEM Jobs", inplace=True)

        data.replace(to_replace={"Marital status": { 
            'single':'alone',
            'divorced':'alone',
            'legally separated':'alone',
            'widower':'alone',
            'married':'together',
            'facto union':'together'
        }}, inplace=True)
                                            
        return data
    
    @staticmethod
    def getDummies(data: pd.DataFrame) -> pd.DataFrame:
        """get dummies

        Args:
            train (`pd.DataFrame`): Train dataframe to be treated

        Returns:
            `pd.DataFrame` : treated dataframe
        """    
        
        data = pd.get_dummies(data=data, prefix_sep="-", dummy_na=True, drop_first=False)

        return data
    
    @staticmethod
    def scaleData(data: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
        """Tranforms the values in the dataframe to fit in a scale of 0 to 1

        Args:
            train (pd.DataFrame): Unscaled train dataframe

        Returns:
            pd.DataFrame: Scaled dataframe
        """    

        scaler = MinMaxScaler()
        scaler.fit(data)

        data = pd.DataFrame(scaler.transform(data), columns = data.columns, index = data.index)

        return data, scaler
    
    @staticmethod
    def encodeSuccess(data: pd.DataFrame) -> pd.DataFrame:
        """Replace string values on success by integers

        Args:
            data (pd.DataFrame): Untreated dataframe 

        Returns:
            pd.DataFrame: Treated dataframe
        """
        data = data.replace({'Success': {"Gave up": 0, "Holding on": 1, "Succeeded": 2}})

        return data
    
    @staticmethod
    def addAverages(data: pd.DataFrame) -> pd.DataFrame:
        """Group 'period' features into one, the average of both

        Args:
            data (pd.DataFrame): Dataframe with 'period' features ungrouped

        Returns:
            pd.DataFrame: Dataframe with 'period' features grouped
        """        
        data['Average grades']=(data['Average grade 1st period']+data['Average grade 2nd period'])/2
        data['Average units taken']=(data['N units taken 1st period']+data['N units taken 2nd period'])/2
        data['Average scored units']=(data['N scored units 1st period']+data['N scored units 2nd period'])/2
        data['Average units approved']=(data['N units approved 1st period']+data['N units approved 2nd period'])/2
        data['Average units credited']=(data['N units credited 1st period']+data['N units credited 2nd period'])/2
        data['Average unscored units']=(data['N unscored units 1st period']+data['N unscored units 2nd period'])/2

        return data
    
    @staticmethod
    def runPreprocessing(
        data: pd.DataFrame, 
        metricFeatures: list[str], 
        boolFeatures: list[str], 
        academicFeatures: list[str], 
        demographicFeatures: list[str],
        removedAcademicFeatures: list[str] | list = [],
        removedDemographicFeatures: list[str] | list = [],
        *,
        grouping: Literal["low", "high"] = "high"
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler, MinMaxScaler]:
        """Runs the preprocessing steps on the dataframe

        Args:
            data (pd.DataFrame): Un-preprocessed data dataframe
            metricFeatures (list[str]): Metric features of the dataframe
            boolFeatures (list[str]): Categorical features of the dataframe
            academicFeatures (list[str]): Features belonging to the academic perspective
            demographicFeatures (list[str]): Features belonging to the demographic perspective
            removedAcademicFeatures (list[str] | list, optional): Features removed from the academic perspective. Defaults to [].
            removedDemographicFeatures (list[str] | list, optional): Features removed from the demographic perspective. Defaults to [].
            grouping (Literal['low', 'high'], optional): Choose a way to group the variables. Defaults to "high".

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler, MinMaxScaler]: Treated dataframes, original, academic perspective & demographic perspective, and the their respective MinMaxScaler objects used to unscale when showing the results
        """        
        data = data.drop('Observations', axis=1).drop_duplicates()
        data = data[data['Registered'] == 'Yes']
        data = data.drop('Registered', axis=1)

        data = Preprocessing.encodeSuccess(data)
        data = Preprocessing.fillNa(data, metricFeatures, boolFeatures)
        data = Preprocessing.addAverages(data)
        data = Preprocessing.removeOutliers(data)
        data = Preprocessing.groupValues(data, grouping)

        dataAcademic: pd.DataFrame = data[academicFeatures]
        dataDemographic: pd.DataFrame = data[demographicFeatures]

        dataAcademic = Preprocessing.getDummies(dataAcademic)
        dataDemographic = Preprocessing.getDummies(dataDemographic)

        dataAcademic = dataAcademic.drop(removedAcademicFeatures, axis=1)
        dataDemographic = dataDemographic.drop(removedDemographicFeatures, axis=1)

        dataAcademic, academicScaler = Preprocessing.scaleData(dataAcademic)
        dataDemographic, demographicScaler = Preprocessing.scaleData(dataDemographic)

        # dataAcademic = dataAcademic[["Entry score", "Average scored units", "Average grades", "Average units approved", "Success"]]

        data = Preprocessing.getDummies(data)
        data, scaler = Preprocessing.scaleData(data)

        return data, dataAcademic, dataDemographic, scaler, academicScaler, demographicScaler

class FeatureSelection:
    @staticmethod
    def pairPlots(data: pd.DataFrame, name: str) -> None:
        """Show pair plots for the dataframe

        Args:
            data (pd.DataFrame): Dataframe to get the pair plot generated for
            name (str): Name to be used in the graph title and output file name
        """        
        sns.pairplot(data.sample(1000)).savefig(f"./output/{name} Pair Plot.png")

    @staticmethod
    def checkCorr(data: pd.DataFrame, corrMethod: Literal["pearson", "kendall", "spearman"] = "spearman") -> pd.DataFrame:
        """Obtain correlation matrices and features that have a correlation above a hardcoded threshold

        Args:
            data (pd.DataFrame): Data to get correlation matrix from
            corrMethod (Literal['pearson', 'kendall', 'spearman'], optional): Correlation method. Defaults to "spearman".

        Returns:
            pd.DataFrame: Dataframe containing the features that have a correlation higher than the threshold
        """        
        mask: np.ndarray | pd.DataFrame | pd.Series
        
        mask = np.zeros_like(data.corr(method=corrMethod))
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(50,50))
            ax = sns.heatmap(data.corr(method=corrMethod),mask=mask, annot = True,cmap='coolwarm',square=True,vmin=-1, vmax=1)

        cor_spearman: pd.DataFrame = data.corr(method=corrMethod)

        pd.set_option('display.max_columns',None)
        pd.set_option('display.max_rows',None)
        # display only highly correlated (>=80%) features
        threshold: float = 0.8

        mask = cor_spearman.abs() > threshold

        high_cor: pd.DataFrame = cor_spearman[mask].stack().reset_index()
        high_cor.columns = ['Feature 1', 'Feature 2', 'Correlation']

        # filter out where Feature1==Feature2
        mask = high_cor['Feature 1'] == high_cor['Feature 2']
        high_cor_filtered: pd.DataFrame = high_cor[~mask]

        return high_cor_filtered
    
    @staticmethod
    def getVariableClusterGraphs(data: pd.DataFrame, label:str) -> None:
        """Shows counts of the values of each feature in each cluster

        Args:
            data (pd.DataFrame): Dataframe to get the data from
            label (str): Column name of the cluster labels
        """        
        for i in data.columns:
            sns.histplot(data, x = i, hue=label, kde = True, legend = True, palette = 'Dark2')
            plt.show()

    @staticmethod
    def clusterProfiles(data: pd.DataFrame, label_columns: list, figsize: tuple[int,int], compar_titles=None):
        """Get cluster profiles

        Args:
            data (pd.DataFrame): Dataframe to get the data from
            label_columns (list): Column names of the cluster labels
            figsize (tuple[int,int]): Size of the figure
            compar_titles (_type_, optional): legit idk what this is???. Defaults to None.
        """
        if compar_titles == None:
            compar_titles = [""]*len(label_columns)

        fig, axes = plt.subplots(nrows=len(label_columns), ncols=2, figsize=figsize, squeeze=False)
        for ax, label, titl in zip(axes, label_columns, compar_titles):
            # Filtering df
            drop_cols = [i for i in label_columns if i!=label]
            dfax = data.drop(drop_cols, axis=1)

            # Getting the cluster centroids and counts
            centroids = dfax.groupby(by=label, as_index=False).mean()
            counts = dfax.groupby(by=label, as_index=False).count().iloc[:,[0,1]]
            counts.columns = [label, "counts"]
            color = sns.color_palette('Dark2')

            # Setting Data
            pd.plotting.parallel_coordinates(centroids, label, color=color, ax=ax[0])
            sns.barplot(x=label, y="counts", data=counts, ax=ax[1], palette = color)

            #Setting Layout
            handles, _ = ax[0].get_legend_handles_labels()
            cluster_labels = ["Cluster {}".format(i) for i in range(len(handles))]
            ax[0].annotate(text=titl, xy=(0.95,1.1), xycoords='axes fraction', fontsize=16, fontweight = 'heavy')
            ax[0].legend(handles, cluster_labels) # Adaptable to number of clusters
            ax[0].axhline(color="black", linestyle="--")
            ax[0].set_title("Cluster Means - {} Clusters".format(len(handles)), fontsize=16)
            ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=-20)
            ax[1].set_xticklabels(cluster_labels)
            ax[1].set_xlabel("")
            ax[1].set_ylabel("Absolute Frequency")
            ax[1].set_title("Cluster Sizes - {} Clusters".format(len(handles)), fontsize=16)


        plt.subplots_adjust(hspace=0.4, top=0.90, bottom = 0.2)
        plt.suptitle("Cluster Profiling", fontsize=23)
        plt.show()
