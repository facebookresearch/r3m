import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

runs_df = pd.read_csv("BC_0210.csv")


# tasks = ["pen-v0","relocate-v0"]
tasks = [
    "kitchen_knob1_on-v3","kitchen_light_on-v3","kitchen_sdoor_open-v3","kitchen_ldoor_open-v3",
    "kitchen_micro_open-v3","assembly-v2-goal-observable","bin-picking-v2-goal-observable",
    "button-press-topdown-v2-goal-observable","drawer-open-v2-goal-observable","hammer-v2-goal-observable",
    "pen-v0","relocate-v0"]

load_paths = ["R3M", "MoCo (345)", "CLIP", "ImageNet", "Random"]
colors = ["#B20D30", "#3F784C", "#C17817", "#3F84E5", "#F0E2E7"]
# load_paths = ["R3M", "R3M (-Aug)", "R3M (-L1)", "R3M (-Lang)"]
# colors = ["#B20D30", "#F38D68", "#662C91", "#17A398"]
nummodels = 13


views = ["v1", "v2", "v3"]


def plot_by_task(df, t):
    dfc = df.groupby(['load_path', "finetune"]).count().reset_index()
    assert((dfc["max_success"] == 9).all())

    dfm = df.groupby(['load_path', "finetune"]).mean().reset_index()
    dfse = df.groupby(['load_path', "finetune"]).std().reset_index() 

    fig, ax = plt.subplots(figsize=(2, 4))
    x = 0
    width = 0.1

    c = [-2, -1, 0, 1, 2]
    for i, p in enumerate(load_paths):
        if p == "Random":
            bl = True
        else:
            bl = False
        dfmp = dfm.loc[dfm['load_path'] == p]
        dfmp = dfmp.loc[dfmp['finetune'] == bl]

        dfsep = dfse.loc[dfse['load_path'] == p]
        dfsep = dfsep.loc[dfsep['finetune'] == bl]

        nms = float(dfmp["max_success"])
        nss = float(dfsep["max_success"]) / np.sqrt(9)

        ax.bar(x - c[i] * width, nms, width, yerr=nss, label=p, edgecolor="white", color=colors[i])
    # plt.legend()
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    fig.tight_layout()
    plt.savefig(f"plots/{task}_exp1.png")
    plt.show()

def plot_all(df):
    numruns = 108
    dfc = df.groupby(['load_path', "finetune"]).count().reset_index()
    assert((dfc["max_success"] == numruns).all())

    dfm = df.groupby(['load_path', "finetune"]).mean().reset_index()
    dfse = df.groupby(['load_path', "finetune"]).std().reset_index() 

    fig, ax = plt.subplots(figsize=(12, 4))
    x = 0
    width = 0.1

    c = [-2, -1, 0, 1, 2]
    for i, p in enumerate(load_paths):
        if p == "Random":
            bl = True
        else:
            bl = False
        dfmp = dfm.loc[dfm['load_path'] == p]
        dfmp = dfmp.loc[dfmp['finetune'] == bl]

        dfsep = dfse.loc[dfse['load_path'] == p]
        dfsep = dfsep.loc[dfsep['finetune'] == bl]

        nms = float(dfmp["max_success"])
        nss = float(dfsep["max_success"]) / np.sqrt(numruns)

        ax.bar(x - c[i] * width, nms, width, yerr=nss, label=p, edgecolor="white", color=colors[i])
    plt.legend(framealpha=1, ncol=5)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    # fig.tight_layout()
    plt.savefig(f"plots/exp1_all.png")
    plt.show()
    
    

# for task in tasks:
#     print(task)
#     print("*"*50)
#     df = runs_df.loc[runs_df['env'] == task]
#     plot_by_task(df, task)

df = runs_df.loc[runs_df['env'].isin(tasks)]
plot_all(df)
    

# print(runs_df)
# for run in runs_df:
#     print(run)
#     assert(False)
