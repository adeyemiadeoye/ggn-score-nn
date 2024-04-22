"""
    Plot settings for:
        Regularized Gauss-Newton for learning overparameterized neural networks.
"""
# credits: https://github.com/adeyemiadeoye/SelfConcordantSmoothOptimization.jl/blob/main/examples/paper/exp-utils-paper.jl

using Plots: plot, plot!, scatter, scatter!, default, pyplot, RGBA, @layout, savefig
using PyPlot: plt
using Measures
pyplot()

markershape = [:dtriangle :rtriangle :rect :diamond :hexagon :xcross :utriangle :ltriangle :pentagon :heptagon :octagon :star4 :star6 :star7 :star8]

_plot_args = Dict{Symbol, Any}(
    :background => :white,
    :framestyle => :box,
    :grid => false,
    :minorgrid => false,
    :gridalpha => 0.2,
    :linewidth => 1.5,
    :thickness_scaling => 2,
    :tickfontsize => 9,
    :gridlinewidth => 0.7,
    :minorgridalpha => 0.06,
    :palette => :Dark2_8,
    :background_color_legend => RGBA(1.0,1.0,1.0,0.5),
    :legendfontsize => 8,
    :size => (550, 400)
)

default(; _plot_args...)

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 17
rcParams["lines.linewidth"] = 2.5
rcParams["lines.linestyle"] = "-."
rcParams["ytick.direction"] = "in"
rcParams["xtick.direction"] = "in"