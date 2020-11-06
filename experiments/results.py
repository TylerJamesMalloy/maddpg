import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import math

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)

folders = ["Spread/gDDPG", "Crypto/gMACL", "Crypto/gMACL"]
coef = "0p01"
Num_Agents = 3
agents = [1,2,3,4,5,6,7,8,9,10]
all_data = pd.DataFrame()

for folder in folders:
    for agent_id in agents:
        print("./" + folder + "/Results/" + "a" + str(agent_id) + "/a" + str(agent_id) + "_" + str(coef) + "_agrewards.pkl")
        if(os.path.isfile("./" + folder + "/Results/" + "a" + str(agent_id) + "/a" + str(agent_id) + "_" + str(coef) + "_agrewards.pkl")):
            data = pd.read_pickle("./" + folder + "/Results/" + "a" + str(agent_id) + "/a" + str(agent_id) + "_" + str(coef) + "_agrewards.pkl")
            print(data)
            assert(False)
            data = np.reshape(data, (-1, Num_Agents))
            for index, data_point in enumerate(data):
                if("gDDPG" in folder):
                    d = {"Episode": index * 100, "Reward": data_point[1] , "Model": "MADDPG", "Environment": folder}
                    all_data = all_data.append(d, ignore_index = True)
                if("gMACL" in folder):
                    d = {"Episode": index * 100, "Reward": data_point[1] , "Model": "MACL", "Environment": folder}
                    all_data = all_data.append(d, ignore_index = True)

#all_data = all_data.loc[all_data["Episode"] > 1100]
#all_data = all_data.loc[all_data["Episode"] < 200000]
print(all_data)

ax = sns.lineplot(x="Episode", y="Reward", hue="Model", data=all_data, ci="sd")  
ax.set_xlabel('Training Episode', fontsize=48)
ax.set_ylabel('Average Reward', fontsize=48)

plt.title('Speaker Listener MultiAgent Training Results', fontsize=64)
plt.show()


