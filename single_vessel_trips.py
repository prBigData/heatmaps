"""
SINGLE VESSEL TRIP PACKAGE :
Every helper we could develop to plot a vessel's trip
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # mandatory to be able to export plots from remote server
import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.basemap import Basemap
from sklearn import svm


def load_pos_data(spark, sqlContext):
    """LOAD DATAS"""

    # load tables and create a view
    positions = sqlContext.read.format(
        "org.apache.spark.sql.cassandra"
    ).options(
        table="basic_position",
        keyspace="ais_datas"
    ).load().createOrReplaceTempView("pos")

    p = spark.sql("SELECT * FROM pos")

    return (p)


def get_vessel_trip(spark, sqlContext, mmsi, after=None, before=None):
    """Get every position of a vessel for the specified params"""
    p = load_pos_data(spark, sqlContext)

    trip = p.filter(p.mmsi == str(mmsi))
    if before:
        trip = trip.filter(p.timestamps <= before)
    if after:
        trip = trip.filter(p.timestamps >= after)

    return trip.collect()


def plot_vessel_trip(spark, sqlContext, mmsi, after=None, before=None,
                     filename="plot", file_path="./plots/"):
    """plots every position of a vessel for the specified params"""
    filename += '_'
    rows = get_vessel_trip(spark, sqlContext, mmsi, after, before)

    data = np.array(
        [[row['longitude'], row['latitude']] for row in rows]
        )

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
    x, y = m(data[:, 0], data[:, 1])
    e = m.scatter(x, y, c='gold', s=10)
    e = m.fillcontinents(color='gainsboro', lake_color="lime")
    # plt.show()

    plt.savefig(file_path + filename + str(mmsi) + '_trip.pdf', format="pdf")
