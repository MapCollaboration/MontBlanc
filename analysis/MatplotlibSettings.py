# Plot settings
import matplotlib.pyplot as mpl

# Style
mpl.style.use("seaborn-talk")

# Text and font
mpl.rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 24})
mpl.rc("text", usetex = True)
mpl.rcParams["text.latex.preamble"] = r"\boldmath"
#mpl.rcParams["text.latex.preamble"] = [r"\usepackage{xcolor}"]
mpl.rcParams["font.weight"] = "bold"

# Axes
mpl.rcParams["axes.linewidth"] = 1.3
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 24

# Ticks
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["xtick.major.width"] = 1.3
mpl.rcParams["xtick.minor.size"] = 5
mpl.rcParams["xtick.minor.width"] = 1.3
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.labelsize"] = 20

mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.major.size"] = 10
mpl.rcParams["ytick.major.width"] = 1.3
mpl.rcParams["ytick.minor.size"] = 5
mpl.rcParams["ytick.minor.width"] = 1.3
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["ytick.labelsize"] = 20

# Lines
mpl.rcParams["lines.linewidth"] = 3
mpl.rcParams["lines.color"] = "red"
mpl.rcParams["lines.markersize"] = 10

# Legend
mpl.rcParams["legend.fontsize"] = 24
mpl.rcParams["legend.frameon"] = False

# Figure
mpl.rcParams["savefig.bbox"] = "tight"

