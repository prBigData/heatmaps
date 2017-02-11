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

# CLEANING TO FINISH

t = spark.sql("SELECT mmsi, timestamps, latitude, longitude FROM vessels WHERE timestamps >= \"2016-12-10 22:53:00\" AND timestamps < \"2016-12-10 22:54:00\" ")
t.createOrReplaceTempView("unique_time")
coo = spark.sql("SELECT latitude, longitude FROM unique_time")
dfLat = coo.rdd.map(lambda l:  l.latitude).collect()
dfLong = coo.rdd.map(lambda l:  l.longitude).collect()
data = np.array([[dfLong[i], dfLat[i]] for i in range(0, len(dfLat))])
rbf_clf = svm.OneClassSVM(nu=0.3, kernel="rbf", gamma=0.2)
rbf_clf.fit(data)
xx, yy = np.meshgrid(np.linspace(-4, 37, 50), np.linspace(26, 48, 50))
rbf_Z = rbf_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
rbf_Z = rbf_Z.reshape(xx.shape)

fig = plt.figure()
map = Basemap(
        projection='gall',
        llcrnrlon=-4,              # lower-left corner longitude
        llcrnrlat=26,               # lower-left corner latitude
        urcrnrlon=37,               # upper-right corner longitude
        urcrnrlat=48,               # upper-right corner latitude
        resolution='l',
        area_thresh=1000.0,
    )

map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='grey', zorder=0)  # alpha=0.5

x, y = map(xx, yy)
datax, datay = map(data[:, 0], data[:, 1])
map.contourf(
    x, y, rbf_Z,
    levels=np.linspace(rbf_Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = map.contour(x, y, rbf_Z, levels=[0], linewidths=2, colors='darkred')
map.contourf(x, y, rbf_Z, levels=[0, rbf_Z.max()], colors='palevioletred')
pt = map.scatter(datax, datay, c='gold', s=20, zorder=1)
plt.title("Density map of the vessels in Mediterranean sea")
plt.savefig('fig/fig14.png')
