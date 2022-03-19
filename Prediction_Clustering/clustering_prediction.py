import warnings

from kneed import KneeLocator
from sklearn.cluster import KMeans

from Logging.logging import Logger
from model_methods.model_methods import modelMethods

warnings.simplefilter(action='ignore', category=FutureWarning)

class ClusterPrediction:

    """
    Description: This method is used to assign a cluster to every observation in the data

    Written By: Shivam Shinde

    Version: 1.0

    Revision: None
    """

    def __init__(self):
        self.logger = Logger()
        self.file_object = open("../PredictionLogs/clusteringLogs.txt", "a+")


    def createElbowPlot(self,data):

        """
        Description: This method is used to create a elbow plot using KMeans clustering algorithm. This method also
        returns the ideal number of cluster for the provided data

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param data: The data which need to be clustered

        :return: Ideal number of clusters
        """

        self.logger.log(self.file_object, "Finding the optimal number of clusters into which the data can be split")

        wcss = []
        try:
            # finding the value of wcss for the number of clusters from 2 to 11
            for i in range(2,11):
                kmeans = KMeans(n_clusters=i, init="k-means++",random_state=345)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)

            # finding the optimal number of clusters for the data
            kn = KneeLocator(range(2,11),wcss, curve="convex",direction="decreasing")
            self.logger.log(self.file_object, f"Optimal number of clusters for the provided data is {kn.knee}")
            return kn.knee

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while finding the optimal number of clusters for "
                                              f"the data. Exception: {str(e)}")
            raise e


    def createCluster(self,data,numOfClusters):

        """
        Description: This method is used to create and assign a cluster number to every observation in the data

        Written By: Shivam Shinde

        Version: 1.0

        Revision: None

        :param data: The data on which the clustering needs to be performed

        :param numOfClusters: Ideal number of clusters into which the data needs to be clustered

        :return: Data having an additional column containing the cluster number for each of the observation in the data
        """

        self.logger.log(self.file_object, "Performing the clustering on the data")
        self.data = data
        try:
            # creating a clustering model for the data
            self.kmeans = KMeans(n_clusters=numOfClusters, init="k-means++", random_state=38497)

            # predicting the cluster number to which every data observation belong to.
            self.y_means = self.kmeans.fit_predict(data)

            # saving the clustering model created
            modelmethods = modelMethods()
            modelmethods.modelSaving(self.kmeans,"KMeansCluster",numOfClusters)

            # adding a column containing cluster number for each of the data observations
            self.data['ClusterNumber'] = self.y_means

            self.logger.log(self.file_object, f"Successfully created {str(numOfClusters)} clusters for the data")

            return self.data

        except Exception as e:
            self.logger.log(self.file_object, f"Exception occurred while clustering the data. Exception: {str(e)}")
            raise e


