import numpy as np
import re

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
fontsize = 9
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.titlepad'] = 7
mpl.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.figsize"] = [4,3]

is_graph = True

take_ln = False
moving_avg = True
save = True
save_val = True
window_size = 2500
dataset_num = 8
mean_from_last = 20000
remove_repeats = True #Caused by starting from the same counter multiple times

scale = 1.0
ratio = 1.3 # 1.618
width = scale * 3.3
height = (width / 1.618)
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

f = plt.figure()

#
#
#
#
#

labels_sets = [
    ["LSTM", "DNC"], #1
    ["Spiral", "LSTM"], #2
    ["", ""],
    ["Full, Gradient Loss", "Full, No Gradient Loss", "Crop, Gradient Loss", "Crop, No Gradient Loss"], #4
    ["With Decay, Edge Penalty 0.05", "Without Decay, Edge Penalty 0.05", "Without Decay, Edge Penalty 0.10", ], #5
    ["Mean Squared Error", "Maximum Region Error"], #6
    ["Exponential Decay", "Cyclic Decay"], #7
    [r"Projection 128 $\to$ 64", "No Projection"], #8
    ["Actor LR 0.0010, Critic LR 0.0010", "Actor LR 0.0003, Critic LR 0.0003", "Actor LR 0.0003, Critic LR 0.0010, Self-Competitive", "Actor LR 0.0003, Critic LR 0.0010"], #9
    ["Edge Penalty 0.0", "Edge Penalty 0.1", "Edge Penalty 0.2", "Edge Penalty 0.4"], #10
    ["Fully Supervised", "Supervised Start", "No Supervision"], #11
    ["Replayed Actions", "Replayed Actions, Doubled Noise", "Live Actions"], #12
    ["Buffer Size 50000, 4 Repeats", "Buffer Size 25000, 4 Repeats", "Buffer Size 10000, 10 Repeats", #13
     "Buffer Size 10000, 4 Repeats", "Buffer Size 10000, 1 Repeats", "Buffer Size 5000, 4 Repeats"], #14
    ["No Normalization, Clipping", "Normalization, Clipping", "Normalization, No Clipping"], #15
    ["LSTM", "GRU", "Fixed Path 1 (Spiral)", "Fixed Path 2 (Actor)", "Fixed Path 3 (Actor)", "Fixed Path 4 (Actor)"], #16
    ["Loss Clipping", "No Loss Clipping"], #17
    ["No Replay Buffer", "Replay Buffer Size 25000", "Replay Buffer Size 100000", "Replay Buffer Size 200000"], #18
    ]

sets = [
    [125, 127],
    [124, 123],
    [119, 122],
    [117, 116, 121, 119],
    [118, 119, 120],
    [114, 115],
    [112, 113],
    [106, 101],
    [100, 101, 99, 98],
    [140, 138, 142, 141],
    [143, 138, 139],
    [134, 136, 123],
    [131, 123, 133, 130, 132, 129],
    [289, 280, 290],
    [280, 283, 145, 285, 286, 287],
    [280, 290],
    [291, 293, 280, 292],
    ]

losses_sets = []
iters_sets = []
for i, (data_nums, labels) in enumerate(zip(sets, labels_sets)):

    if i != 11:
        continue

    print(f"######################## Graph {i+1} ########################")

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labeltop=False, labelright=False)
    for j, dataset_num in enumerate(data_nums):

        log_loc = f"Z:/Jeffrey-Ede/models/recurrent_conv-1/{dataset_num}/"
        
        if is_graph:
            mse_indicator = "generator_losses"
            log_file = log_loc+"log.txt"
            val_file = log_loc+"val_log.txt"

            notes_file = log_loc+"notes.txt"
            with open(notes_file, "r") as f:
                for l in f: print(l)

            val_mode = labels[j] == "Quartic, Clipped, Val"

            switch = False
            losses = []
            vals = []
            losses_iters = []
            with open(log_file, "r") as f:

                numbers = []
                for line in f:
                    numbers += line.split(",")

                numbers = [re.findall(r"([-+]?\d*\.\d+|\d+)", x)[0] for x in numbers if mse_indicator in x]
                losses = [float(x) for x in numbers]
                losses_iters = [i for i in range(1, len(losses)+1)]

            def moving_average(a, n):
                #a = np.concatenate([a[0]*np.ones((n)),a, a[0]*np.ones((n))])
                ret = np.cumsum(np.insert(a,0,0), dtype=float)
                return (ret[n:] - ret[:-n]) / float(n)

            #if i in [13]:
            #    losses_iters = losses_iters[:500_000]
            #    losses = losses[:len(losses_iters)]
            
            avg_size = window_size if val_mode else window_size

            losses = moving_average(np.array(losses[:]), avg_size) if moving_avg else np.array(losses[:])
            losses_iters = moving_average(np.array(losses_iters[:]), avg_size) if moving_avg else np.array(losses[:])
        
            print(np.mean((losses[(len(losses)-mean_from_last):])[np.isfinite(losses[(len(losses)-mean_from_last):])]))
            
            losses_iters = [i/100_000 for i in losses_iters]
            losses_sets.append(losses)
            iters_sets.append(losses_iters)

            val_losses_file = log_loc + "val_losses.npy"

            plot_label = labels[j]
            if os.path.isfile(val_losses_file):
                val_loss = np.load(val_losses_file)
                print(val_loss)
                val_loss = val_loss[val_loss < 4]
                print(labels[j], np.mean(val_loss), np.std(val_loss))

                plot_label += f". Test: {np.mean(val_loss):.3f}, {np.std(val_loss):.3f}"
            else:
                print(labels[j], "NA")


            plt.plot(losses_iters, losses if take_ln else losses, label=plot_label, linewidth=1)
        else:
            val_losses_file = log_loc + "val_losses.npy"

            if os.path.isfile(val_losses_file):
                val_loss = np.load(val_losses_file)
                print(val_loss)
                val_loss = val_loss[val_loss < 4]
                print(labels[j], np.mean(val_loss), np.std(val_loss))
            else:
                print(labels[j], "NA")



    plt.xlabel('Training Iterations (x10$^5$)')
    plt.ylabel('Mean Squared Error')

    plt.legend(loc='upper right', frameon=False, fontsize=8)

    plt.minorticks_on()

    name = str(i+1)
    save_loc =  "Z:/Jeffrey-Ede/models/recurrent_conv-1/" + name + ".png"
    print(save_loc)
    plt.savefig( save_loc, bbox_inches='tight', )

    plt.gcf().clear()
