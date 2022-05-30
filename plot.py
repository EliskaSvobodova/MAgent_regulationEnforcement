"""plot general log file according to given indexes"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    indexes = [1, 2, 5, 7, 10, 12,16]
    filenames = ["logs/exp2__" + str(index) + ".log" for index in indexes]
    plot_filename = "logs/exp_2.png"
    log_values = ["reward"]
    agent_types = ["compliant", "defective"]
    
    total_data = []
    for filename in filenames:
        data = []
        with open(filename, 'r') as fin:
            for line in fin.readlines():
                items = line.split('\t')

                row = []
                for item in items[2:4]:
                    t = eval(item.split(':')[1])
                    if isinstance(t, list):
                        for x in t:
                            row.append(x)
                    else:
                        row.append(t)
                if len(row) > 0:
                    data.append(row)
        data = np.array(data)
        print(data.shape)
        total_data.append(data)
        
    # data.col = loss[n_agent_types], reward[n_agent_types], value[n_agent_types]
    total_data = np.array(total_data)
    print(total_data.shape)
    num = 50

    plt.figure()
    means = []
    for agent_idx, agent_type in enumerate(agent_types):
        d = total_data[:,:, agent_idx]
        means = np.sum(d[:,-num:], axis=1) / num
        print(means.shape)
        plt.plot(indexes, means, label=agent_type)
        plt.xlabel("tau")
        plt.ylabel("reward")
        # plt.plot(d, alpha=0.6, label=agent_type)
        plt.plot()
    plt.legend()
    # plt.title(f"{title} - {', '.join([f'{a}: {m: .2f}' for a, m in zip(agent_types, means)])}")

    plt.tight_layout()
    plt.savefig(plot_filename)
