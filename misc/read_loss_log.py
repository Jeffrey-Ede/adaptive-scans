import re
import matplotlib.pyplot as plt
import numpy as np

from collections.abc import Iterable

import os

take_ln = False
moving_avg = True
save = True 
save_val = True
window_size = 1_000
#dataset_nums = [158, 159, 160, 161, 163] #162 not run yet
#dataset_nums = [152, 207, 209, 211, 213, 214, 215, 216]
#dataset_nums = [277]
dataset_nums = [125, 145, 149, 280, 293, 294, 295]
#dataset_nums = [125, 145, 269, 271, 272, 273, 274, 275]
if not isinstance(dataset_nums, Iterable):
    dataset_nums = [dataset_nums]

rand_walk = False #True
mean_from_last = 20_000
remove_repeats = True #Caused by starting from the same counter multiple times
clip_val = 25

save = None #"Z:/Jeffrey-Ede/models/phd_thesis/"

for dataset_num in list(dataset_nums):

    log_loc = r"Z:\Jeffrey-Ede\models\recurrent_conv-1" + "\\"
    log_file = log_loc + f"{dataset_num}\\" + f"log.txt"
    val_file = log_loc + "val_log.txt"

    #mse_indicator = "actor_losses"
    #mse_indicator = "critic_losses"
    mse_indicator = "generator_losses"

    notes_file = log_loc+f"{dataset_num}\\notes.txt"
    with open(notes_file, "r") as f:
        for i, l in enumerate(f):
            if not i:
                print(f"{dataset_num}:", l)
            else:
                print(l)

    val_losses_file = log_loc + f"{dataset_num}\\val_losses.npy"
    if os.path.isfile(val_losses_file):
        val_loss = np.load(val_losses_file)
        val_loss = val_loss[val_loss < 4]
        #val_loss = np.exp(np.mean(np.log(val_loss)))
        val_loss = np.mean(val_loss)
        print("Validation Losses:", val_loss)
    else:
        print("Validation Losses: NA")

    switch = False
    losses = []
    vals = []
    losses_iters = []
    with open(log_file, "r") as f:

        numbers = []
        for line in f:
            numbers += line.split(",")

        #print(numbers)
        #print(len(numbers))
        #vals = [re.findall(r"([-+]?\d*\.\d+|\d+)", x)[0] for x in numbers if "Val" in x]
        numbers = [re.findall(r"([-+]?\d*\.\d+|\d+)", x)[0] for x in numbers if mse_indicator in x]
        #print(numbers)
        #print(vals)#; print(numbers)
        #print(numbers)
        losses = [min(float(x), clip_val) for x in numbers]
        #print(losses)
        losses_iters = [i for i in range(1, len(losses)+1)]
    try:
        switch = False
        val_losses = []
        val_iters = []
        with open(val_file, "r") as f:
            for line in f:
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                for i in range(1, len(numbers), 2):
                    val_losses.append(float(numbers[i]))
                    val_iters.append(float(numbers[i-1]))
    except:
        pass

    def moving_average(a, n=window_size):
        ret = np.cumsum(np.insert(a,0,0), dtype=float)
        return (ret[n:] - ret[:-n]) / float(n)

    #if dataset_num == 75:
    #    losses_iters = losses*np.array(losses_iters)

    losses = moving_average(np.array(losses[:])) if moving_avg else np.array(losses[:])
    losses_iters = moving_average(np.array(losses_iters[:])) if moving_avg else np.array(losses[:])
    val_losses = moving_average(np.array(val_losses[:])) if moving_avg else np.array(val_losses[:])
    val_iters = moving_average(np.array(val_iters[:])) if moving_avg else np.array(val_iters[:])

    print(np.mean((losses[(len(losses)-mean_from_last):-3000])[np.isfinite(losses[(len(losses)-mean_from_last):-3000])]))

    #if save:
    #    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
    #                str(dataset_num)+"/log.npy")
    #    np.save(save_loc, losses)

    #if save_val:
    #    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
    #                str(dataset_num)+"/val_log.npy")
    #    np.save(save_loc, val_losses)

    #    save_loc = ("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-"+
    #                str(dataset_num)+"/val_iters.npy")
    #    np.save(save_loc, val_iters)

    #print(losses)
    #print(losses_iters)

    if save:
        np.save(save+f"learning_rate_optimization-{dataset_num}_iters.npy", losses_iters)
        np.save(save+f"learning_rate_optimization-{dataset_num}_losses.npy", losses)

    plt.plot(losses_iters, np.log(losses) if take_ln else losses, label=f"{dataset_num}")
    #plt.plot(val_iters, np.log(val_losses) if take_ln else val_losses)

plt.legend()
plt.show()

