# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:39:08 2021

@author: pccom
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

plot_dir = os.path.dirname(os.path.abspath(__file__))

markersize = 15
linewidth = 3
fontsize = 20
fontsize_legend = 18
marker_edge_width = 3
tick_fontsize = 20


framealpha_val=0.0
borderpad_val =0.1
labelspacing_val = 0.3
handlelength_val = 1.0
handletextpad_val = 0.5
borderaxespad_val = 0.2

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Times"]})
rc("text", usetex=True)
matplotlib.rcParams.update(
    {"text.usetex": True, "text.latex.preamble": [r"\usepackage{newtxmath}"]}
)


def plotting_function_self_concordance_paper(
    list_primal_gaps,
    list_times,
    list_markers,
    list_colors,
    list_labels,
    list_labels_x_axis,
    label_y_axis,
    list_y_limits,
    list_legend,
    filename,
    location_legend=["best", "best"],
):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10, 4))
    max_iteration = max([len(v) for v in list_primal_gaps])

    for i in range(len(list_primal_gaps)):
        axes[0].loglog(
            np.arange(len(list_primal_gaps[i])) + 1,
            list_primal_gaps[i],
            marker=list_markers[i],
            color=list_colors[i],
            linewidth=linewidth,
            markevery=(
                np.logspace(0, np.log10(len(list_primal_gaps[i]) - 2), 10).astype(int)
                - 1
            ).tolist(),
            markerfacecolor="none",
            markeredgewidth=marker_edge_width,
            markersize=markersize,
            label=list_labels[i],
        )

    # axes[0].grid()
    # axes[0].legend(ncol=1, fontsize = fontsize_legend)
    axes[0].set_ylabel(label_y_axis, fontsize=fontsize)
    axes[0].set_xlabel(list_labels_x_axis[0], fontsize=fontsize)
    axes[0].set_xlim([1, max_iteration])
    axes[0].set_ylim(list_y_limits[0])
    axes[0].tick_params(axis="x", labelsize=tick_fontsize)
    axes[0].tick_params(axis="y", labelsize=tick_fontsize)
    if list_legend[0]:
        axes[0].legend(
            ncol=1,
            fontsize=fontsize_legend,
            framealpha=framealpha_val,
            loc=location_legend[0],
            borderpad=borderpad_val,
            labelspacing = labelspacing_val,
            handlelength = handlelength_val,
            handletextpad = handletextpad_val,
            borderaxespad = borderaxespad_val,
        )

    max_time = max([v[-1] for v in list_times])

    for i in range(len(list_primal_gaps)):
        axes[1].loglog(
            list_times[i],
            list_primal_gaps[i],
            marker=list_markers[i],
            color=list_colors[i],
            linewidth=linewidth,
            markevery=(
                np.logspace(0, np.log10(len(list_primal_gaps[i]) - 2), 10).astype(int)
                - 1
            ).tolist(),
            markerfacecolor="none",
            markeredgewidth=marker_edge_width,
            markersize=markersize,
            label=list_labels[i],
        )
    # axes[1].grid()
    axes[1].set_xlabel(list_labels_x_axis[1], fontsize=fontsize)
    axes[1].set_xlim([None, max_time])
    axes[0].set_ylim(list_y_limits[1])
    axes[1].tick_params(axis="x", labelsize=tick_fontsize)
    axes[1].tick_params(axis="y", labelsize=tick_fontsize)
    if list_legend[1]:
        axes[1].legend(
            ncol=1,
            fontsize=fontsize_legend,
            framealpha=framealpha_val,
            loc=location_legend[1],
            borderpad=borderpad_val,
            labelspacing = labelspacing_val,
            handlelength = handlelength_val,
            handletextpad = handletextpad_val,
            borderaxespad = borderaxespad_val,
        )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    plt.savefig(
        os.path.join(plot_dir, "..", "Images", filename),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    
    ###################################Birkhoff polytope EXPERIMENT############################################
    filepath = os.path.join(plot_dir, "..", "results", "syn_1000_1200_10_50_1.mat_birkhoff.json")
    import json
    
    with open(filepath) as f:
        data = json.load(f)
    
    # 0 - Iteration
    # 1 - Primal value
    # 2 - Primal - Dual value
    # 3 - Frank-Wolfe
    # 4 - Time
    
    # Get agnostic algorithm values
    primal_gap_agnostic = np.zeros(len(data["agnostic"]))
    FW_gap_agnostic = np.zeros(len(data["agnostic"]))
    time_agnostic = np.zeros(len(data["agnostic"]))
    for i in range(len(data["agnostic"])):
        primal_gap_agnostic[i] = data["agnostic"][i][1] - data["awaystep"][-1][1]
        FW_gap_agnostic[i] = data["agnostic"][i][3]
        time_agnostic[i] = data["agnostic"][i][4] - data["agnostic"][0][4]
    
    # Get backtrack algorithm values
    primal_gap_awaystep = np.zeros(len(data["awaystep"]))
    FW_gap_awaystep = np.zeros(len(data["awaystep"]))
    time_awaystep = np.zeros(len(data["awaystep"]))
    for i in range(len(data["awaystep"])):
        primal_gap_awaystep[i] = data["awaystep"][i][1] - data["awaystep"][-1][1]
        FW_gap_awaystep[i] = data["awaystep"][i][3]
        time_awaystep[i] = data["awaystep"][i][4] - data["awaystep"][0][4]
    primal_gap_awaystep[0] = primal_gap_agnostic[0]
    FW_gap_awaystep[0] = FW_gap_agnostic[0]
        
    # Get backtrack algorithm values
    primal_gap_backtracking = np.zeros(len(data["backtracking"]))
    FW_gap_backtracking = np.zeros(len(data["backtracking"]))
    time_backtracking = np.zeros(len(data["backtracking"]))
    for i in range(len(data["backtracking"])):
        primal_gap_backtracking[i] = data["backtracking"][i][1] - data["awaystep"][-1][1]
        FW_gap_backtracking[i] = data["backtracking"][i][3]
        time_backtracking[i] = data["backtracking"][i][4] - data["backtracking"][0][4]
    
    # Get monotonous algorithm values
    primal_gap_monotonous = np.zeros(len(data["monotonous"]))
    FW_gap_monotonous = np.zeros(len(data["monotonous"]))
    time_monotonous = np.zeros(len(data["monotonous"]))
    for i in range(len(data["monotonous"])):
        primal_gap_monotonous[i] = data["monotonous"][i][1] - data["awaystep"][-1][1]
        FW_gap_monotonous[i] = data["monotonous"][i][3]
        time_monotonous[i] = data["monotonous"][i][4] - data["monotonous"][0][4]
        
    list_primal_gaps = [
        primal_gap_monotonous,
        primal_gap_agnostic,
        primal_gap_backtracking,
        primal_gap_awaystep,
    ]
    list_times = [time_monotonous, time_agnostic, time_backtracking, time_awaystep]
    list_markers = ["^", "o", "D", "s", '*', 'p']
    list_colors = ["g", "k", "c", "r", "b", "m"]
    list_labels = [r"\texttt{M-FW}", r"\texttt{FW}", r"\texttt{B-FW}", r"\texttt{B-AFW}", r"\texttt{GSC-FW}", r"\texttt{LLOO}"]
    list_labels_x_axis = [r"Iteration", r"Time [s]"]
    label_y_axis_primal = r"$h(\mathbf{x}_{t})$"
    list_y_limits = [(None, None), (None, None)]
    plotting_function_self_concordance_paper(
        list_primal_gaps,
        list_times,
        list_markers,
        list_colors,
        list_labels,
        list_labels_x_axis,
        label_y_axis_primal,
        list_y_limits,
        [False, False],
        "Primal_gap_Birkhoff.pdf",
    )
    
    list_dual_gaps = [
        FW_gap_monotonous,
        FW_gap_agnostic,
        FW_gap_backtracking,
        FW_gap_awaystep,
    ]
    label_y_axis_dual = r"$g(\mathbf{x}_{t})$"
    list_y_limits = [(None, None), (None, None)]
    plotting_function_self_concordance_paper(
        list_dual_gaps,
        list_times,
        list_markers,
        list_colors,
        list_labels,
        list_labels_x_axis,
        label_y_axis_dual,
        list_y_limits,
        [False, True],
        "Dual_gap_Birkhoff.pdf",
    )
    
    
    ###################################KL divergence EXPERIMENT############################################
    filepath = os.path.join(plot_dir, "..", "results", "data_2000_1000_50.mat.json")
    import json
    
    with open(filepath) as f:
        data = json.load(f)
    
    # 0 - Iteration
    # 1 - Primal value
    # 2 - Primal - Dual value
    # 3 - Frank-Wolfe
    # 4 - Time
    
    # Get agnostic algorithm values
    primal_gap_agnostic = np.zeros(len(data["agnostic"]))
    FW_gap_agnostic = np.zeros(len(data["agnostic"]))
    time_agnostic = np.zeros(len(data["agnostic"]))
    for i in range(len(data["agnostic"])):
        primal_gap_agnostic[i] = data["agnostic"][i][1] - data["awaystep"][-1][1]
        FW_gap_agnostic[i] = data["agnostic"][i][3]
        time_agnostic[i] = data["agnostic"][i][4] - data["agnostic"][0][4]
    
    # Get backtrack algorithm values
    primal_gap_awaystep = np.zeros(len(data["awaystep"]))
    FW_gap_awaystep = np.zeros(len(data["awaystep"]))
    time_awaystep = np.zeros(len(data["awaystep"]))
    for i in range(len(data["awaystep"])):
        primal_gap_awaystep[i] = data["awaystep"][i][1] - data["awaystep"][-1][1]
        FW_gap_awaystep[i] = data["awaystep"][i][3]
        time_awaystep[i] = data["awaystep"][i][4] - data["awaystep"][0][4]
        
    primal_gap_awaystep = primal_gap_awaystep[:300]
    FW_gap_awaystep = FW_gap_awaystep[:300]
    time_awaystep = time_awaystep[:300]
    
    # Get backtrack algorithm values
    primal_gap_backtracking = np.zeros(len(data["backtracking"]))
    FW_gap_backtracking = np.zeros(len(data["backtracking"]))
    time_backtracking = np.zeros(len(data["backtracking"]))
    for i in range(len(data["backtracking"])):
        primal_gap_backtracking[i] = data["backtracking"][i][1] - data["awaystep"][-1][1]
        FW_gap_backtracking[i] = data["backtracking"][i][3]
        time_backtracking[i] = data["backtracking"][i][4] - data["backtracking"][0][4]
    
    # Get monotonous algorithm values
    primal_gap_monotonous = np.zeros(len(data["monotonous"]))
    FW_gap_monotonous = np.zeros(len(data["monotonous"]))
    time_monotonous = np.zeros(len(data["monotonous"]))
    for i in range(len(data["monotonous"])):
        primal_gap_monotonous[i] = data["monotonous"][i][1] - data["awaystep"][-1][1]
        FW_gap_monotonous[i] = data["monotonous"][i][3]
        time_monotonous[i] = data["monotonous"][i][4] - data["monotonous"][0][4]
        
    # Get monotonous algorithm values
    primal_gap_second = np.zeros(len(data["second_order"]))
    FW_gap_second = np.zeros(len(data["second_order"]))
    time_second = np.zeros(len(data["second_order"]))
    for i in range(len(data["second_order"])):
        primal_gap_second[i] = data["second_order"][i][1] - data["awaystep"][-1][1]
        FW_gap_second[i] = data["second_order"][i][3]
        time_second[i] = data["second_order"][i][4] - data["second_order"][0][4]
        
        
    list_primal_gaps = [
        primal_gap_monotonous,
        primal_gap_agnostic,
        primal_gap_backtracking,
        primal_gap_awaystep,
        primal_gap_second,
    ]
    list_times = [time_monotonous, time_agnostic, time_backtracking, time_awaystep, time_second]
    list_markers = ["^", "o", "D", "s", '*', 'p']
    list_colors = ["g", "k", "c", "r", "b", "m"]
    list_labels = [r"\texttt{M-FW}", r"\texttt{FW}", r"\texttt{B-FW}", r"\texttt{B-AFW}", r"\texttt{GSC-FW}", r"\texttt{LLOO}"]
    list_labels_x_axis = [r"Iteration", r"Time [s]"]
    label_y_axis_primal = r"$h(\mathbf{x}_{t})$"
    list_y_limits = [(None, None), (None, None)]
    plotting_function_self_concordance_paper(
        list_primal_gaps,
        list_times,
        list_markers,
        list_colors,
        list_labels,
        list_labels_x_axis,
        label_y_axis_primal,
        list_y_limits,
        [False, False],
        "Primal_gap_KL.pdf",
    )
    
    list_dual_gaps = [
        FW_gap_monotonous,
        FW_gap_agnostic,
        FW_gap_backtracking,
        FW_gap_awaystep,
        FW_gap_second,
    ]
    label_y_axis_dual = r"$g(\mathbf{x}_{t})$"
    list_y_limits = [(None, None), (None, None)]
    plotting_function_self_concordance_paper(
        list_dual_gaps,
        list_times,
        list_markers,
        list_colors,
        list_labels,
        list_labels_x_axis,
        label_y_axis_dual,
        list_y_limits,
        [False, True],
        "Dual_gap_KL.pdf",
    )
    
    ###################################Portfolio EXPERIMENT############################################
    # filepath = os.path.join(os.getcwd(), "..", "results", "synlog_5000_2000.mat.json")
    # filepath = os.path.join(os.getcwd(), "..", "results", "syn_1000_800_10_50_1.mat.json")
    filepath = os.path.join(plot_dir, "..", "results", "syn_1000_1200_10_50.mat_second_order.json")
    
    import json
    
    with open(filepath) as f:
        data = json.load(f)
    
    # 0 - Iteration
    # 1 - Primal value
    # 2 - Primal - Dual value
    # 3 - Frank-Wolfe
    # 4 - Time
    
    # Get agnostic algorithm values
    primal_gap_agnostic = np.zeros(len(data["agnostic"]))
    FW_gap_agnostic = np.zeros(len(data["agnostic"]))
    time_agnostic = np.zeros(len(data["agnostic"]))
    for i in range(len(data["agnostic"])):
        primal_gap_agnostic[i] = data["agnostic"][i][1] - data["away_step"][-1][1]
        FW_gap_agnostic[i] = data["agnostic"][i][3]
        time_agnostic[i] = data["agnostic"][i][4] - data["agnostic"][0][4]
    
    # Get backtrack algorithm values
    primal_gap_awaystep = np.zeros(len(data["away_step"]))
    FW_gap_awaystep = np.zeros(len(data["away_step"]))
    time_awaystep = np.zeros(len(data["away_step"]))
    for i in range(len(data["away_step"])):
        primal_gap_awaystep[i] = data["away_step"][i][1] - data["away_step"][-1][1]
        FW_gap_awaystep[i] = data["away_step"][i][3]
        time_awaystep[i] = data["away_step"][i][4] - data["away_step"][0][4]
    
    primal_gap_awaystep = primal_gap_awaystep[:60]
    FW_gap_awaystep = FW_gap_awaystep[:60]
    time_awaystep = time_awaystep[:60]
    
    # Get backtrack algorithm values
    primal_gap_backtracking = np.zeros(len(data["backtrack"]))
    FW_gap_backtracking = np.zeros(len(data["backtrack"]))
    time_backtracking = np.zeros(len(data["backtrack"]))
    for i in range(len(data["backtrack"])):
        primal_gap_backtracking[i] = data["backtrack"][i][1] - data["away_step"][-1][1]
        FW_gap_backtracking[i] = data["backtrack"][i][3]
        time_backtracking[i] = data["backtrack"][i][4] - data["backtrack"][0][4]
    
    # Get monotonous algorithm values
    primal_gap_monotonous = np.zeros(len(data["monotonous"]))
    FW_gap_monotonous = np.zeros(len(data["monotonous"]))
    time_monotonous = np.zeros(len(data["monotonous"]))
    for i in range(len(data["monotonous"])):
        primal_gap_monotonous[i] = data["monotonous"][i][1] - data["away_step"][-1][1]
        FW_gap_monotonous[i] = data["monotonous"][i][3]
        time_monotonous[i] = data["monotonous"][i][4] - data["monotonous"][0][4]
    primal_gap_monotonous = primal_gap_monotonous[:480]
    FW_gap_monotonous = FW_gap_monotonous[:480]
    time_monotonous = time_monotonous[:480]
  
    # Get second-order algorithm values
    primal_gap_second = np.zeros(len(data["second_order"]))
    FW_gap_second = np.zeros(len(data["second_order"]))
    time_second = np.zeros(len(data["second_order"]))
    for i in range(len(data["second_order"])):
        primal_gap_second[i] = data["second_order"][i][1] - data["away_step"][-1][1]
        FW_gap_second[i] = data["second_order"][i][3]
        time_second[i] = data["second_order"][i][4] - data["second_order"][0][4]
 
    filepath_LLOO = os.path.join(plot_dir, "..", "results", "syn_1000_1200_10_50.mat_second_order_lloo.json")
    with open(filepath_LLOO) as f:
        data_LLOO = json.load(f)   
 
    # Get LLOO algorithm values
    primal_gap_LLOO = np.zeros(len(data_LLOO["lloo"]))
    FW_gap_LLOO = np.zeros(len(data_LLOO["lloo"]))
    time_LLOO = np.zeros(len(data_LLOO["lloo"]))
    for i in range(len(data_LLOO["lloo"])):
        primal_gap_LLOO[i] = data_LLOO["lloo"][i]['primal'] - data["away_step"][-1][1] - 12.385135127290818
        FW_gap_LLOO[i] = data_LLOO["lloo"][i]['dual_gap']
        time_LLOO[i] = data_LLOO["lloo"][i]['time'] - data_LLOO["lloo"][0]['time']
       
 
    list_primal_gaps = [
        primal_gap_monotonous,
        primal_gap_agnostic,
        primal_gap_backtracking,
        primal_gap_awaystep,
        primal_gap_second,
        primal_gap_LLOO,
    ]
    list_times = [time_monotonous, time_agnostic, time_backtracking, time_awaystep, time_second, time_LLOO]
    list_y_limits = [(None, None), (None, None)]
    plotting_function_self_concordance_paper(
        list_primal_gaps,
        list_times,
        list_markers,
        list_colors,
        list_labels,
        list_labels_x_axis,
        label_y_axis_primal,
        list_y_limits,
        [False, False],
        "Primal_gap_Portfolio.pdf",
    )
    
    list_dual_gaps = [
        FW_gap_monotonous,
        FW_gap_agnostic,
        FW_gap_backtracking,
        FW_gap_awaystep,
        FW_gap_second,
        FW_gap_LLOO,
    ]
    list_y_limits = [(None, None), (None, None)]
    plotting_function_self_concordance_paper(
        list_dual_gaps,
        list_times,
        list_markers,
        list_colors,
        list_labels,
        list_labels_x_axis,
        label_y_axis_dual,
        list_y_limits,
        [False, True],
        "Dual_gap_Portfolio.pdf",
    )
    
    ###################################LogReg EXPERIMENT############################################
    filepath = os.path.join(plot_dir, "..", "results", "a4a.csv.json")
    import json
    
    with open(filepath) as f:
        data = json.load(f)
    
    # 0 - Iteration
    # 1 - Primal value
    # 2 - Primal - Dual value
    # 3 - Frank-Wolfe
    # 4 - Time
    
    # Get agnostic algorithm values
    primal_gap_agnostic = np.zeros(len(data["agnostic"]))
    FW_gap_agnostic = np.zeros(len(data["agnostic"]))
    time_agnostic = np.zeros(len(data["agnostic"]))
    for i in range(len(data["agnostic"])):
        primal_gap_agnostic[i] = data["agnostic"][i][1] - data["awaystep"][-1][1]
        FW_gap_agnostic[i] = data["agnostic"][i][3]
        time_agnostic[i] = data["agnostic"][i][4] - data["agnostic"][0][4]
    
    # Get backtrack algorithm values
    primal_gap_awaystep = np.zeros(len(data["awaystep"]))
    FW_gap_awaystep = np.zeros(len(data["awaystep"]))
    time_awaystep = np.zeros(len(data["awaystep"]))
    for i in range(len(data["awaystep"])):
        primal_gap_awaystep[i] = data["awaystep"][i][1] - data["awaystep"][-1][1]
        FW_gap_awaystep[i] = data["awaystep"][i][3]
        time_awaystep[i] = data["awaystep"][i][4] - data["awaystep"][0][4]
    
    
    # Get backtrack algorithm values
    primal_gap_backtracking = np.zeros(len(data["backtracking"]))
    FW_gap_backtracking = np.zeros(len(data["backtracking"]))
    time_backtracking = np.zeros(len(data["backtracking"]))
    for i in range(len(data["backtracking"])):
        primal_gap_backtracking[i] = data["backtracking"][i][1] - data["awaystep"][-1][1]
        FW_gap_backtracking[i] = data["backtracking"][i][3]
        time_backtracking[i] = data["backtracking"][i][4] - data["backtracking"][0][4]
    
    # Get monotonous algorithm values
    primal_gap_monotonous = np.zeros(len(data["monotonous"]))
    FW_gap_monotonous = np.zeros(len(data["monotonous"]))
    time_monotonous = np.zeros(len(data["monotonous"]))
    for i in range(len(data["monotonous"])):
        primal_gap_monotonous[i] = data["monotonous"][i][1] - data["awaystep"][-1][1]
        FW_gap_monotonous[i] = data["monotonous"][i][3]
        time_monotonous[i] = data["monotonous"][i][4] - data["monotonous"][0][4]
    
    
    filepath_LLOO = os.path.join(plot_dir, "..", "results", "a4a.csv_second_order.json")
    with open(filepath_LLOO) as f:
        data_second = json.load(f)   
 
    # Get LLOO algorithm values
    primal_gap_second  = np.zeros(len(data_second["second_order"]))
    FW_gap_second  = np.zeros(len(data_second["second_order"]))
    time_second  = np.zeros(len(data_second["second_order"]))
    for i in range(len(data_second["second_order"])):
        primal_gap_second[i] = data_second["second_order"][i][1] - data["awaystep"][-1][1]
        FW_gap_second[i] = data_second["second_order"][i][3]
        time_second[i] = data_second["second_order"][i][4] - data_second["second_order"][0][4]

    
    list_primal_gaps = [
        primal_gap_monotonous,
        primal_gap_agnostic,
        primal_gap_backtracking,
        primal_gap_awaystep,
        primal_gap_second,
    ]
    list_times = [time_monotonous, time_agnostic, time_backtracking, time_awaystep, time_second]
    list_y_limits = [(None, None), (None, None)]
    plotting_function_self_concordance_paper(
        list_primal_gaps,
        list_times,
        list_markers,
        list_colors,
        list_labels,
        list_labels_x_axis,
        label_y_axis_primal,
        list_y_limits,
        [False, False],
        "Primal_gap_Logreg.pdf",
    )
    
    list_dual_gaps = [
        FW_gap_monotonous,
        FW_gap_agnostic,
        FW_gap_backtracking,
        FW_gap_awaystep,
        FW_gap_second,
    ]
    list_y_limits = [(None, None), (None, None)]
    plotting_function_self_concordance_paper(
        list_dual_gaps,
        list_times,
        list_markers,
        list_colors,
        list_labels,
        list_labels_x_axis,
        label_y_axis_dual,
        list_y_limits,
        [True, False],
        "Dual_gap_Logreg.pdf",
        location_legend=["upper right", "upper right"],
    )
