class Histogram():
    def __init__(self):
        self.rwidth = None
        self.elinewidth = None
        self.orientation = None
        self.axis = None
        self.lthick = None
        self.linestyle = None
        self.linecolor = None
        self.barcolor = None
        self.nbins = None
        self.es = None
        self.hasErr = None

class Line():
    def __init__(self):
        self.hasMarker = None
        self.elinewidth = None
        self.marker = None
        self.markers = None
        self.lthick = None
        self.lthicks = None
        self.linestyle = None
        self.linestyles = None
        self.marker_size = None
        self.marker_sizes = None
        self.linecolor = None
        self.linecolors = None
        # plots
        self.prob_same_x = None


class Scatter():
    def __init__(self):
        self.marker = None
        self.markers = None
        self.marker_size = None
        self.marker_sizes = None
        self.colorbar_side = None
        self.colorbar_size = None
        self.colorbar_pad = None
        self.elinewidth = None

        # self.hasMarker = None
        # self.lthick = None
        # self.lthicks = None
        # self.linestyle = None
        # self.linestyles = None
        # self.linecolor = None
        # self.linecolors = None
        # # plots
        # self.prob_same_x = None