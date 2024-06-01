import sompy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


class Clustering:
    @staticmethod
    def getSomDetails(data: pd.DataFrame, rows: int=25, cols: int=25) -> tuple[sompy.sompy.SOM, int, int, np.float32]:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            rows (int, optional): _description_. Defaults to 25.
            cols (int, optional): _description_. Defaults to 25.

        Returns:
            tuple[sompy.sompy.SOM, int, int, np.float32]: _description_
        """
        df_som = np.float32(data.values)
        mapsize: list[int] = [rows, cols]

        som: sompy.sompy.SOM = sompy.sompy.SOMFactory().build(df_som, mapsize, mask=None,
                                mapshape='planar', # 2Dimensions
                                lattice='rect', # topology: 'rect' or 'hexa'
                                normalization='var', # standardize the variables
                                initialization='pca', # initialization of the weights: 'pca' or 'random'
                                neighborhood='gaussian', # neighborhood function: 'gaussian' or 'bubble'
                                training='batch') # training mode: 'seq' or 'batch'
        
        return som,rows,cols,df_som

    @staticmethod
    def getGraphs(data: pd.DataFrame, somDetails: tuple[sompy.sompy.SOM, int, int, np.float32], name: str) -> None:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            somDetails (tuple[sompy.sompy.SOM, int, int, np.float32]): _description_
            name (str): _description_
        """
        somDetails[0].train(n_job=1, verbose=None, train_rough_len=3, train_finetune_len=5) # type: ignore

        u = sompy.umatrix.UMatrixView(width=somDetails[1], height=somDetails[2], title=f'{name} U-matrix', show_axis=True, text_size=8, show_text=True)

        #This is the Umat value
        UMAT  = u.build_u_matrix(som=somDetails[0], distance=1, row_normalized=False)

        #Here you have Umatrix plus its render
        _, umat = u.show(som=somDetails[0], distance=1, row_normalized=True, contour=True, blob=False)

        somDetails[0].component_names = data.columns
        comp_planes = sompy.mapview.View2DPacked(width=somDetails[1], height=somDetails[2], title=f'{name} Component Planes', text_size=8)
        comp_planes.show(somDetails[0], what='codebook', which_dim='all', col_sz=8)

    @staticmethod
    def clustering(data: pd.DataFrame, somDetails: tuple[sompy.sompy.SOM, int, int, np.float32],name: str, nClusters: int = 0) -> tuple[sompy.hitmap.HitMapView, pd.DataFrame]:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            somDetails (tuple[sompy.sompy.SOM, int, int, np.float32]): _description_
            name (str): _description_
            nClusters (int, optional): _description_. Defaults to 0.

        Returns:
            tuple[sompy.hitmap.HitMapView, pd.DataFrame]: _description_
        """        
        somDetails[0].cluster(n_clusters=nClusters)
        labels = getattr(somDetails[0], 'cluster_labels')
        h = sompy.hitmap.HitMapView(10, 10, f'{name} Hitmap', text_size=8, show_text=True)
        h.show(somDetails[0], )
        bmus = somDetails[0].project_data(somDetails[3])
        data['bmu'] = bmus
        data['label'] = labels[data['bmu']]

        clusteringResult: pd.DataFrame = data.groupby(['label']).describe().T

        return h, clusteringResult

    @staticmethod
    def somWrapper(data: pd.DataFrame, name: str, nClusters: int = 0) -> tuple[sompy.hitmap.HitMapView, pd.DataFrame] | None:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            name (str): _description_
            nClusters (int, optional): _description_. Defaults to 0.

        Returns:
            tuple[sompy.hitmap.HitMapView, pd.DataFrame] | None: _description_
        """        
        somDetails: tuple[sompy.sompy.SOM, int, int, np.float32] = Clustering.getSomDetails(data)
        Clustering.getGraphs(data, somDetails, name)
        if nClusters != 0:
            return Clustering.clustering(data, somDetails, name, nClusters)
        
    @staticmethod
    def clusterProfiles(data, label_columns, figsize, compar_titles=None):
        """_summary_

        Args:
            data (_type_): _description_
            label_columns (_type_): _description_
            figsize (_type_): _description_
            compar_titles (_type_, optional): _description_. Defaults to None.
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