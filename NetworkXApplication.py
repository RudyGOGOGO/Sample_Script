import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
# step 2: load the network data and get the Dataframe object
def getDegreeCDFData(dataPath):
    print("=======Start Calculation=======")
    data = pd.read_csv(dataPath)
    print("*******Data Set Overview*******")
    print(data.info())
    # step 3: generate the network data
    network_data = nx.from_pandas_edgelist(data, source='source', target='target', edge_attr='type')
    print("*******Graph Overview*******")
    print(nx.info(network_data))
    print('Density: {0}'.format(nx.density(network_data)))
    # generate the statistics for the graph: top-10 institutes
    betweenness = pd.Series(nx.betweenness_centrality(network_data), name='Betweenness')
    company_company = pd.Series.to_frame(betweenness)
    company_company['Closeness'] = pd.Series(nx.closeness_centrality(network_data))
    company_company['PageRank'] = pd.Series(nx.pagerank_scipy(network_data))
    company_company['Degree'] = pd.Series(dict(nx.degree(network_data)))
    desc_betweenness = company_company.sort_values('Betweenness', ascending=False)
    print(desc_betweenness.head(10))
    # step 4: get the degree distribution sequence
    degree = nx.degree_histogram(network_data)
    x = list(range(len(degree)))
    y = [z / float(sum(degree)) for z in degree]
    # remove the <x, 0> pairs
    indices_with_zero_value = [i for i, one in enumerate(y) if one == 0]
    y_without_zero = list(filter(lambda a: a != 0, y))
    # delete them in reverse order so that you don't throw off the subsequent indexes
    for index in sorted(indices_with_zero_value, reverse=True):
        del x[index]

    np_arr_x = np.asarray(x, dtype=float)
    np_arr_y = np.asarray(y_without_zero, dtype=float)
    logX = np.log10(np_arr_x).reshape(-1, 1)
    logY = np.log10(np_arr_y).reshape(-1, 1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(logX, logY)  # perform linear regression
    logY_pred = linear_regressor.predict(logX)  # make predictions
    print("=======End Calculation=======")

    return logX, logY, logY_pred

def plotScatterAndRegression(x, y, y_pred):
    plt.plot(x, y, "*")
    plt.plot(x, y_pred, "b")
    plt.xlabel("Log(K)")
    plt.ylabel("Log(P(K))")
    plt.show()


if __name__ == '__main__':
    x, y, y_pred = getDegreeCDFData("/Users/weizhang/Desktop/NetworkAnalysis/2016-2-utf8.csv")
    # x, y, y_pred = getDegreeCDFData(sys.argv[1])
    plotScatterAndRegression(x, y, y_pred)