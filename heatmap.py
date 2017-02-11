""" HEATMAP SCRIPT

Generates an heatmap with every position recorded between two dates
"""

import numpy as np
import matplotlib
import datetime
matplotlib.use('Agg')  # mandatory to be able to export plots from server
# yet to be removed when working on your own machine
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.basemap import Basemap
from sklearn import svm


def load_data(spark, sqlContext):
    """pre-load data"""

    vessels = sqlContext.read.format(
            "org.apache.spark.sql.cassandra"
        ).options(
            table="basic_position",
            keyspace="ais_datas"
        ).load().createOrReplaceTempView("vessels")

    v = spark.sql("SELECT * FROM vessels")

    return v


def get_all(spark, sqlContext, before, after):
    """get any point between two timestamps"""

    v = load_data(spark, sqlContext)

    p = v.filter(
        v.timestamps >= after
    ).filter(
        v.timestamps < before
    ).collect()

    # report values : 2016-12-10 22:53:00 < ts < 2016-12-10 22:54:00
    return p


def plot_all(spark, sqlContext, before, after, filename="plot",
             file_path="./plots/"):
    """plot any detected vessel position between two timestamps"""

    filename += '_'
    rows = get_all(spark, sqlContext, before, after)

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
        file_path + filename + str(before) + '_' + str(after) + '_svm.pdf',
        format="pdf")
