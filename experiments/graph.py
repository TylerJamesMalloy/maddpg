import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import math

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)


folders = [ "Spread/gDDPG/Models/", 
            "Spread/gMACL/Models_0p001/", 
            "Spread/gMACL/Models_0p0001/", 
]

"""
            "Adversary/gMACL/Models_0p001/",
            "Adversary/gMACL/Models_0p0001/",
            "Adversary/gDDPG/Models_0p001/",
            "Adversary/gDDPG/Models_0p0001/",
            "Speaker_Listener/gDDPG/Models/", 
            "Speaker_Listener/gMACL/Models_0p001/", 
            "Speaker_Listener/gMACL/Models_0p0001/", 
            "Speaker_Listener/gMACL/Models_0p00001/",  
            "Push/gMACL/Models_0p001/",
            "Push/gMACL/Models_0p0001/",
            "Push/gDDPG/Models_0p001/",
            "Push/gDDPG/Models_0p0001/",          
            "Adversary/gMACL/Models_0p001/",
            "Adversary/gMACL/Models_0p0001/",
            "Adversary/gDDPG/Models_0p001/",
            "Adversary/gDDPG/Models_0p0001/",
"""
coefs = ["0", "0p001", "0p0001", "0p00001"]
Num_Agents = 3
agents = [1,2,3,4,5]
all_data = pd.DataFrame()

for folder in folders:
    for agent_id in agents:
        for coef in coefs:
            if(os.path.isfile("./" + folder  + "a" + str(agent_id) + "/a" + str(agent_id) + "_" + str(coef) + "_agrewards.pkl")):
                data = pd.read_pickle("./" + folder + "a" + str(agent_id) + "/a" + str(agent_id) + "_" + str(coef) + "_agrewards.pkl")
                data = np.reshape(data, (-1, Num_Agents))
                for index, data_point in enumerate(data):
                    if("gDDPG" in folder):
                        d = {"Episode": int(index /5) * 500, "Reward": data_point[1] , "Model": "MADDPG", "Environment": folder.split("/")[0]}
                        all_data = all_data.append(d, ignore_index = True)
                    if("gMACL" in folder):
                        d = {"Episode": int(index /5) * 500, "Reward": data_point[1] , "Model": "CL-MA " + coef + " vs Adv MADDPG", "Environment": folder.split("/")[0]}
                        all_data = all_data.append(d, ignore_index = True)

#all_data = all_data.loc[all_data["Episode"] > 1000]
#all_data = all_data.loc[all_data["Episode"] < 200000]
print(all_data)

#Speaker_Listener_Data = all_data.loc[all_data["Environment"] == "Speaker_Listener"]
#Push_Data = all_data.loc[all_data["Environment"] == "Push"]

ax = sns.lineplot(x="Episode", y="Reward", hue="Model", data=all_data, ci=99)  
ax.set_xlabel('Training Episode', fontsize=48)
ax.set_ylabel('Good Agent Reward', fontsize=48)
#ax.set_ylim(-45,-13)

plt.title('Adversary Training Results', fontsize=64)
plt.show()


