import sompy
import numpy as np
import pandas as pd

class Clustering:
    @staticmethod
    def getSom(data) -> tuple[sompy.sompy.SOM, int, int, np.float32]:
        """_summary_

        Args:
            data (_type_): _description_

        Returns:
            tuple[sompy.sompy.SOM, int, int, np.float32]: _description_
        """    
        df_som = np.float32(data.values)
        rows = 25
        cols = 25
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
    def getGraphs(data: pd.DataFrame, somDetails: tuple[sompy.sompy.SOM, int, int, np.float32]) -> None:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            somDetails (tuple[sompy.sompy.SOM, int, int, np.float32]): _description_
        """    
        somDetails[0].train(n_job=1, verbose=None, train_rough_len=3, train_finetune_len=5) # type: ignore

        u = sompy.umatrix.UMatrixView(somDetails[1], height=somDetails[2], title='U-matrix', show_axis=True, text_size=8, show_text=True)

        #This is the Umat value
        UMAT  = u.build_u_matrix(somDetails[0], distance=1, row_normalized=False)

        #Here you have Umatrix plus its render
        _, umat = u.show(somDetails[0], distance=1, row_normalized=True, contour=True, blob=False)

        somDetails[0].component_names = data.columns
        comp_planes = sompy.mapview.View2DPacked(somDetails[1], somDetails[2], 'Component Planes', text_size=8)
        comp_planes.show(somDetails[0], what='codebook', which_dim='all', col_sz=8)

    @staticmethod
    def clustering(data: pd.DataFrame, somDetails: tuple[sompy.sompy.SOM, int, int, np.float32], nClusters: int = 4) -> tuple[sompy.hitmap.HitMapView, pd.DataFrame]:
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            somDetails (tuple[sompy.sompy.SOM, int, int, np.float32]): _description_
            clusters (int, optional): _description_. Defaults to 4.
        """    
        somDetails[0].cluster(n_clusters=nClusters)
        labels = getattr(somDetails[0], 'cluster_labels')
        h = sompy.hitmap.HitMapView(10, 10, 'Hitmap', text_size=8, show_text=True)
        h.show(somDetails[0], )
        bmus = somDetails[0].project_data(somDetails[3])
        data['bmu'] = bmus
        data['label'] = labels[data['bmu']]

        clusteringResult: pd.DataFrame = data.groupby(['label']).describe().T

        return h, clusteringResult

    @staticmethod
    def somWrapper(data, getClusters: bool = True, nClusters: int = 4) -> tuple[sompy.hitmap.HitMapView, pd.DataFrame] | None:
        somDetails: tuple[sompy.sompy.SOM, int, int, np.float32] = Clustering.getSom(data)
        Clustering.getGraphs(data, somDetails)
        if getClusters:
            return Clustering.clustering(data, somDetails, nClusters)