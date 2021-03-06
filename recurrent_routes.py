"""
RECURRENT ROUTES :
Every helper we could develop to plot the recurrent routes

from abnormal_routes import *
"""

import numpy as np
import json
import datetime
import matplotlib
matplotlib.use('Agg')  # mandatory to be able to export plots from remote server
# (yet to be removed when working on your own machine)
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.basemap import Basemap
from sklearn import svm


def load_dest_data(spark, sqlContext):
    """initial data loading function"""

    # load tables and create a view
    positions = sqlContext.read.format(
        "org.apache.spark.sql.cassandra"
    ).options(
        table="basic_position",
        keyspace="ais_datas"
    ).load().createOrReplaceTempView("pos")

    destinations = sqlContext.read.format(
        "org.apache.spark.sql.cassandra"
    ).options(
        table="basic_destination",
        keyspace="ais_datas"
    ).load().createOrReplaceTempView("dest")

    # not collected yet
    d = spark.sql("SELECT * FROM dest")
    p = spark.sql("SELECT * FROM pos")

    return (d, p)


def get_all_to_dest(spark, sqlContext, destination):
    """makes the link between destination and position tables"""

    d, p = load_dest_data(spark, sqlContext)

    # select every mmsi - day couple going to the destination
    dest = d.filter(d.destination == destination).collect()
    rows = list()

    # for each mmsi-day, get the positions of that mmsi that day
    for row in dest:
        mmsi = row['mmsi']
        date_to = row['timestamps'].date()
        date_from = row['timestamps'].date() - datetime.timedelta(days=1)
        rows += p.filter(
            p.mmsi == mmsi
        ).filter(
            p.timestamps >= date_from
        ).filter(
            p.timestamps < date_to
        ).collect()

    return rows


def plot_all_to_dest(spark, sqlContext, destination, filename="plot",
                     file_path="./plots/"):
    """Use SVM to approximate a density map of the previously retrieved
    vessels, going to the specified destination"""

    filename += '_'
    rows = get_all_to_dest(spark, sqlContext, destination)

    # SVM
    # Format data the way we need
    data = np.array(
        [[row['longitude'], row['latitude']] for row in rows]
        )

    # Apply SVM
    OCS = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=1)
    OCS.fit(data)

    # divide 2D space into subdivisions
    xx, yy = np.meshgrid(np.linspace(-4, 37, 50), np.linspace(26, 48, 50))

    # get the decision boundary on that space subdivision
    Z = OCS.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot
    m = Basemap(
        projection='gall',
        llcrnrlon=-4,              # lower-left corner longitude
        llcrnrlat=26,               # lower-left corner latitude
        urcrnrlon=37,               # upper-right corner longitude
        urcrnrlat=48,               # upper-right corner latitude
        resolution='l',
        area_thresh=1000.0,
        )
    e = m.drawcoastlines()
    e = m.drawcountries()
    e = m.drawmapboundary(fill_color='steelblue')
    x, y = m(xx, yy)  # mandatory
    e = m.contourf(x, y, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    e = m.contour(x, y, Z, levels=[0], linewidths=2, colors='darkred')
    e = m.contourf(x, y, Z, levels=[0, Z.max()], colors='palevioletred')
    x, y = m(data[:, 0], data[:, 1])  # mandatory
    e = m.scatter(x, y, c='gold', s=10)
    e = m.fillcontinents(color='gainsboro', lake_color="lime")
    # plt.show()

    plt.savefig(
        file_path + filename + str(destination) + '_svm.pdf',
        format="pdf")
