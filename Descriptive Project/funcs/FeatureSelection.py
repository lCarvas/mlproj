import pandas as pd
import seaborn as sns

class FeatureSelection:
    @staticmethod
    def pairPlots(data: pd.DataFrame,fileName: str) -> None:
        sns.pairplot(data.sample(1000)).savefig(f"./output/{fileName}.png")