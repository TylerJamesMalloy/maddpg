import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import math

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)


folders = [ "Data/Mahjong"] 

#coefs = ["1e-1", "1e-2", "1e-3", "1e-4"]
coefs = ["1e-4"]
Num_Agents = 2
agents = [1,2,3,4,5]
all_data = pd.DataFrame()

for folder in folders:
    for agent_id in agents:
        for coef in coefs:
            print("./" + folder + "/" + coef + "/" + "a" + str(agent_id) + "/a" + str(agent_id) +  "_agrewards.pkl")
            if(os.path.isfile("./" + folder + "/" + coef + "/" + "a" + str(agent_id) + "/a" + str(agent_id) +  "_agrewards.pkl")):
                data = pd.read_pickle("./" + folder + "/" + coef + "/" + "a" + str(agent_id) + "/a" + str(agent_id) +   "_agrewards.pkl")
                data = np.reshape(data, (-1, Num_Agents))
                for index, data_point in enumerate(data):
                    d = {"Episode": int(index /10) * 10, "Reward": data_point[0] , "Model": "MADDPG ", "Environment": folder.split("/")[0]}
                    all_data = all_data.append(d, ignore_index = True)

                    d = {"Episode": int(index /10) * 10, "Reward": data_point[1] , "Model": "MACL " + coef, "Environment": folder.split("/")[0]}
                    all_data = all_data.append(d, ignore_index = True)

all_data = all_data.loc[all_data["Episode"] > 500]
#all_data = all_data.loc[all_data["Episode"] < 200000]
print(all_data)

#Speaker_Listener_Data = all_data.loc[all_data["Environment"] == "Speaker_Listener"]
#Push_Data = all_data.loc[all_data["Environment"] == "Push"]

ax = sns.lineplot(x="Episode", y="Reward", hue="Model", data=all_data, ci=99)  
ax.set_xlabel('Training Episode', fontsize=48)
ax.set_ylabel('Average Reward', fontsize=48)
#ax.set_ylim(-30,-5)

plt.title('Competative Keep Away', fontsize=48)
plt.show()


