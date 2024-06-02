import sklearn
import sklearn.preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sompy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import seaborn as sns

sns.set_theme()


class Clustering:
    @staticmethod
    def getSomDetails(data: pd.DataFrame, rows: int = 25, cols: int = 25) -> tuple[sompy.sompy.SOM, int, int]:
        """Get som object, rows and columns of the som

        Args:
            data (pd.DataFrame): Input Dataframe
            rows (int, optional): Number of rows of the som. Defaults to 25.
            cols (int, optional): Number of columns of the som. Defaults to 25.

        Returns:
            tuple[sompy.sompy.SOM, int, int, np.float32]: Return the som, rows and cols of the som
        """
        df_som = np.float32(data) # type: ignore love type hinting
        mapsize: list[int] = [rows, cols]

        som: sompy.sompy.SOM = sompy.sompy.SOMFactory().build(df_som, mapsize, mask=None,
                                mapshape='planar', # 2Dimensions
                                lattice='rect', # topology: 'rect' or 'hexa'
                                normalization='var', # standardize the variables
                                initialization='pca', # initialization of the weights: 'pca' or 'random'
                                neighborhood='gaussian', # neighborhood function: 'gaussian' or 'bubble'
                                training='batch') # training mode: 'seq' or 'batch'
        
        som.train(n_job=1, verbose=None, train_rough_len=3, train_finetune_len=5) # type: ignore

        return som, rows, cols

    @staticmethod
    def getSomGraphs(data: pd.DataFrame, somDetails: tuple[sompy.sompy.SOM, int, int], name: str) -> None:
        """Get graphs pertaining to som

        Args:
            data (pd.DataFrame): Input dataframe
            somDetails (tuple[sompy.sompy.SOM, int, int, np.float32]): Som details
            name (str): Name to be used in the graphs
        """
        u = sompy.umatrix.UMatrixView(width=somDetails[2], height=somDetails[1], title=f'{name} U-matrix', show_axis=True, text_size=8, show_text=True)

        #This is the Umat value
        UMAT  = u.build_u_matrix(som=somDetails[0], distance=1, row_normalized=False)

        #Here you have Umatrix plus its render
        _, umat = u.show(som=somDetails[0], distance=1, row_normalized=True, contour=True, blob=False)

        somDetails[0].component_names = data.columns
        comp_planes = sompy.mapview.View2DPacked(width=somDetails[1], height=somDetails[2], title=f'{name} Component Planes', text_size=8)
        comp_planes.show(somDetails[0], what='codebook', which_dim='all', col_sz=8)

    @staticmethod
    def sompyClustering(
        data: pd.DataFrame,
        somDetails: tuple[sompy.sompy.SOM, int, int],
        nClusters: int = 0,
        ) -> pd.DataFrame:
        """Clusters the values using sompy

        Args:
            data (pd.DataFrame): Dataframe to be clustered
            somDetails (tuple[sompy.sompy.SOM, int, int, np.float32]): Som details
            nClusters (int, optional): Number of clusters to be obtained. Defaults to 0.

        Returns:
            pd.DataFrame: Clustered dataframe
        """     
        bmus = somDetails[0].project_data(data)
        somDetails[0].cluster(n_clusters=nClusters)
        labels = getattr(somDetails[0], 'cluster_labels')
        data['bmu'] = bmus
        data['label'] = labels[data['bmu']]
        data.drop("bmu", axis=1, inplace=True)

        return data

    @staticmethod
    def somWrapper(data: pd.DataFrame, name: str, nClusters: int = 0) -> pd.DataFrame:
        """Organizes the som related functions and runs the necessary ones

        Args:
            data (pd.DataFrame): Input dataframe
            name (str): Name for the graphs
            nClusters (int, optional): Number of clusters to be obtained. Defaults to 0.

        Returns:
            pd.DataFrame: Either the original dataframe or the clustered dataframe, depending on if nClusters is 0 or larger, respectively.
        """    
        somDetails: tuple[sompy.sompy.SOM, int, int] = Clustering.getSomDetails(data)
        if nClusters != 0:
            # print(data.shape, somDetails[0])
            return Clustering.sompyClustering(data, somDetails, nClusters)
        Clustering.getSomGraphs(data, somDetails, name)
        return data
        
    @staticmethod
    def kmeansGraphs(data: pd.DataFrame, elbowGraph: bool = True, silhouetteGraph: bool = True, dendrogram: bool = True) -> None:
        """Graphs pertaining to kmeans

        Args:
            data (pd.DataFrame): Input dataframe
            elbowGraph (bool, optional): Whether or not to display the elbow graph. Defaults to True.
            silhouetteGraph (bool, optional): Whether or not to display the silhouette scores graph. Defaults to True.
            dendrogram (bool, optional): Whether or not to display the dendrogram. Defaults to True.
        """        
        if elbowGraph:
            ks = range(1, 20)
            inertias: list[float] = []

            for k in ks:
                model = KMeans(n_clusters=k, random_state=66)
                model.fit(data)
                inertias.append(model.inertia_)

            plt.plot(ks, inertias)
            plt.xlabel('k')
            plt.ylabel('SSD to cluster center')
            plt.xticks(ks)
            plt.show()

        if silhouetteGraph:
            ks = range(2, 21)
            sil_score: list[float] = []

            for k in ks:
                model = KMeans(n_clusters=k, random_state=66)
                model.fit_predict(data)
                sil_score.append(silhouette_score(data, model.labels_, metric='euclidean')) # type: ignore --- love type hinting s_score returns Float, go check declaration, float | float16 | float32 | float64 but is incompatible with float icant with this bro

            plt.plot(ks, sil_score)
            plt.xlabel('k')
            plt.ylabel('Sillhouette Score')
            plt.xticks(ks)
            plt.show()

        if dendrogram:
            clusters = hierarchy.linkage(data, method="ward")

            plt.figure(figsize=(8, 6))
            hierarchy.dendrogram(clusters)
            plt.show()
    
    @staticmethod
    def runKMeans(data: pd.DataFrame, nClusters: int = 0, *, elbowGraph: bool = True, silhouetteGraph: bool = True, dendrogram: bool = True) -> pd.DataFrame:
        """Runs kmeans clustering if nClusters > 0, else gets the graphs relevant to kmeans

        Args:
            data (pd.DataFrame): Input dataframe
            nClusters (int, optional): Number of clusters to be obtained. Defaults to 0.
            elbowGraph (bool, optional): Whether or not to display the elbow graph. Defaults to True.
            silhouetteGraph (bool, optional): Whether or not to display the silhouette scores graph. Defaults to True.
            dendrogram (bool, optional): Whether or not to display the dendrogram. Defaults to True.

        Returns:
            pd.DataFrame: Either the original dataframe or the clustered dataframe, depending on if nClusters is 0 or larger, respectively. 
        """        
        if nClusters == 0:
            Clustering.kmeansGraphs(data, elbowGraph, silhouetteGraph, dendrogram)
            return data
        
        kmeans: KMeans = KMeans(n_clusters = nClusters, random_state = 100).fit(data)
        data['label'] = kmeans.predict(data)
        
        return data
    
    @staticmethod
    def mergePerspectives(data: pd.DataFrame, dataAcademic: pd.DataFrame, dataDemographic: pd.DataFrame, scaler: sklearn.preprocessing.MinMaxScaler) -> pd.DataFrame:
        """Merges both perspectives

        Args:
            data (pd.DataFrame): Original unseparated dataframe
            dataAcademic (pd.DataFrame): Academic perspective dataframe
            dataDemographic (pd.DataFrame): Demographic perspective dataframe
            scaler (sklearn.preprocessing.MinMaxScaler): MinMaxScaler object of the original dataframe

        Returns:
            pd.DataFrame: Final clustered dataframe
        """        
        data = pd.DataFrame(scaler.inverse_transform(data), index=data.index, columns=data.columns)
        data['academic_profile']=dataAcademic['label']
        data['demographic_profile']=dataDemographic['label']
        data['final_groups'] = data.groupby(['academic_profile', 'demographic_profile'], sort = False).ngroup()
        
        
        unusedColumns: set[str] = set([col for col in data.columns if col not in dataDemographic.columns])
        unusedColumns2: set[str] = set([col for col in data.columns if col not in dataAcademic.columns])

        unusedColumns.difference(unusedColumns2)
        unusedColumns2.difference(unusedColumns)

        usedColumnsAll: set[str] = unusedColumns.difference(unusedColumns2).union(unusedColumns2.difference(unusedColumns))

        unusedColumnsAll = set(data.columns.difference(usedColumnsAll)) # type: ignore cant be bothered
        unusedColumnsAll.remove('final_groups')

        return data.drop(unusedColumnsAll,axis=1) # type: ignore cant be bothered 2 electric boogaloo
        
