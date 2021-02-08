import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import math

plt.style.use('fivethirtyeight')
sns.set(style="ticks", color_codes=True, rc={"lines.linewidth": 2.5})
sns.set(font_scale=2.5)


folders = [ 
            "Data/Leduc/0p01/",]

agents = ['a1','a2','a3','a4']
all_data = pd.DataFrame()

for folder in folders:
    for agent in agents:
        data_file = "./" + folder + agent + "/" + agent + "_agrewards.pkl"
        if(os.path.isfile(data_file)):
            data = pd.read_pickle(data_file)
            data = np.reshape(data, (-1, 2))
            for index, data_point in enumerate(data):
                rounded_index = int(index / 10) * 10
                d = {"Episode": rounded_index, "Reward":  data_point[0] , "Model": "MADDPG " + folder.split("/")[2], "Environment": folder.split("/")[1]}
                all_data = all_data.append(d, ignore_index = True)
                d = {"Episode": rounded_index, "Reward":  data_point[1],  "Model": "CL-MADDPG " + folder.split("/")[2] , "Environment": folder.split("/")[1]}
                all_data = all_data.append(d, ignore_index = True)
        

#all_data = all_data.loc[all_data["Episode"] > 500]
print(all_data)

ax = sns.lineplot(x="Episode", y="Reward", hue="Model", data=all_data, ci="sd")  
ax.set_xlabel('Training Episode', fontsize=48)
ax.set_ylabel('Average Reward', fontsize=48)

plt.title('Uno Card Game', fontsize=48)
plt.show()


