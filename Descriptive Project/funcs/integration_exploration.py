import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import seaborn as sns

sns.set_theme()


class Integration:
    @staticmethod
    def importdata(path: str) -> pd.DataFrame:
        dataDF: pd.DataFrame = pd.read_excel(path + 'data/Descriptive_Data.xlsx')
        dfSuccess: pd.DataFrame = pd.read_excel(path+'data/Descriptive_Data.xlsx',sheet_name=2)
        dataDF = dataDF.set_index('Userid')
        dfSuccess = dfSuccess.set_index('Userid')
        dataDF = dataDF.join(dfSuccess, 'Userid')
        
        return dataDF

class Exploration:
    @staticmethod
    def describeData(data: pd.DataFrame, metricFeatures: list[str], categoricalFeatures: list[str]) -> pd.DataFrame:
        print(f"Duplicaded: {data.duplicated().sum()}\nMissing: {data.isna().sum().sum()}\nNon-Registered (empty): {(data["Registered"] != "Yes").sum()}")
        display(Markdown("### Value Counts"))
        for variable in categoricalFeatures:
            print(data[variable].value_counts())

        display(Markdown("\n\n### Percentage of missing values per column"))
        print(round(data.isnull().sum() / data.shape[0] * 100.00,2))

        data = data.drop('Observations', axis=1).drop_duplicates()
        data = data[data['Registered'] == 'Yes']
        data = data.drop('Registered', axis=1)

        for i, col in enumerate(metricFeatures):
            plt.figure(i)
            sns.boxplot(x=col, data=data)

        return data.describe()
    