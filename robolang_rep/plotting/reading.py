import pandas as pd 
import wandb

api = wandb.Api()
entity, project = "surajnfb", "bc"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 

names = ["BC_0210", "BC_0223"]
summary_list, config_list, name_list = [], [], []
i = 0
everythingdict = {
    "load_path": [],
    "finetune": [],
    "env": [],
    "num_demos" : [],
    "camera": [],
    "seed": [],
    "max_success": []
}

def translatepath(p):
    if p == "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-28/10_agent.finetunelang=0,agent.l1weight=1e-05,agent.langtype=lorel,agent.langweight=1.0,agent.size=50,batch_size=16,dataset=ego4d,doaug=rctraj,experiment=rep_0124/snapshot_1000000.pt":
        return "R3M"
    elif p == "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-28/11_agent.finetunelang=0,agent.l1weight=0.0,agent.langtype=lorel,agent.langweight=1.0,agent.size=50,batch_size=16,dataset=ego4d,doaug=rctraj,experiment=rep_0124/snapshot_1000000.pt":
        return "R3M (-L1)"
    elif p == "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-28/2_agent.finetunelang=0,agent.l1weight=1e-05,agent.langtype=lorel,agent.langweight=1.0,agent.size=50,batch_size=16,dataset=all,doaug=rctraj,experiment=rep_0124/snapshot_760000.pt":
        return "R3M (+SthSth/RoboNet)"
    elif p == "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-28/8_agent.finetunelang=0,agent.l1weight=1e-05,agent.langtype=lorel,agent.langweight=1.0,agent.size=50,batch_size=16,dataset=ego4d,doaug=none,experiment=rep_0124/snapshot_1000000.pt":
        return "R3M (-Aug)"
    elif p == "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-44/2_agent.finetunelang=0,agent.l1weight=1e-05,agent.langtype=lorel,agent.langweight=0.0,agent.size=50,batch_size=16,dataset=all,doaug=rctraj,experiment=rep_0124/snapshot_1000000.pt":
        return "R3M (-Lang)(+SthSth/RoboNet)"
    elif p == "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-44/4_agent.finetunelang=0,agent.l1weight=1e-05,agent.langtype=lorel,agent.langweight=0.0,agent.size=50,batch_size=16,dataset=ego4d,doaug=none,experiment=rep_0124/snapshot_1000000.pt":
        return "R3M (-Lang)(-Aug)"
    elif p == "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-44/6_agent.finetunelang=0,agent.l1weight=1e-05,agent.langtype=lorel,agent.langweight=0.0,agent.size=50,batch_size=16,dataset=ego4d,doaug=rctraj,experiment=rep_0124/snapshot_1000000.pt":
        return "R3M (-Lang)"
    elif p == "/checkpoint/surajn/drqoutput/train_representation/2022-01-24_11-10-44/7_agent.finetunelang=0,agent.l1weight=0.0,agent.langtype=lorel,agent.langweight=0.0,agent.size=50,batch_size=16,dataset=ego4d,doaug=rctraj,experiment=rep_0124/snapshot_1000000.pt":
        return "R3M (-Lang)(-L1)"
    elif p == "clip":
        return "CLIP"
    elif p == "moco345":
        return "MoCo (345)"
    elif p == "":
        return "ImageNet"
    elif p == "random":
        return "Random"

def translatecam(cam):
    if cam in ["left_cap2", "view_1"]:
        return "v1"
    elif cam in ["right_cap2", "view_4"]:
        return "v2"
    elif cam in ["default", "top_cap2", "top"]:
        return "v3"


for run in runs: 
    if run.name not in names:
        continue
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    try:
        everythingdict["max_success"].append(run.summary._json_dict["max_success"])
        everythingdict["load_path"].append(translatepath(run.config["load_path"]))
        everythingdict["seed"].append(run.config["seed"])
        everythingdict["finetune"].append(run.config["finetune"])
        everythingdict["camera"].append(translatecam(run.config["camera"]))
        everythingdict["env"].append(run.config["env"])
        everythingdict["num_demos"].append(run.config["num_demos"])
        everythingdict["name"].append(run.name)
    except:
        pass

    # summary_list.append(run.summary._json_dict)
    # # print(run.summary._json_dict)
    # for key in run.summary._json_dict:
    #     print(key)

    # # .config contains the hyperparameters.
    # #  We remove special values that start with _.
    # config_list.append(
    #     {k: v for k,v in run.config.items()
    #      if not k.startswith('_')})
    # for key in run.config:
    #     print(key)
    # # print(config_list)

    # # .name is the human-readable name of the run.
    # name_list.append(run.name)
    # # print(run.name)
    # # assert(False)
    print(i)
    # assert(False)
    i += 1

runs_df = pd.DataFrame(everythingdict)
dfc = runs_df.groupby(['load_path', "finetune", "env"]).count().reset_index()
print(dfc)

runs_df.to_csv("2seeds.csv")