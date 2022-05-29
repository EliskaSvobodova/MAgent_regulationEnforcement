"""plot general log file according to given indexes"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    filename = "../examples/regulation_enf.log"
    plot_filename = "../regulation_enf.png"
    log_values = ["loss", "reward", "value"]
    agent_types = ["compliant", "defective"]

    data = []

    with open(filename, 'r') as fin:
        for line in fin.readlines():
            items = line.split('\t')

            row = []
            for item in items[1:]:
                t = eval(item.split(':')[1])
                if isinstance(t, list):
                    for x in t:
                        row.append(x)
                else:
                    row.append(t)
            if len(row) > 0:
                data.append(row)
    # data.col = loss[n_agent_types], reward[n_agent_types], value[n_agent_types]
    data = np.array(data)

    fig, axs = plt.subplots(len(log_values), 1)
    for idx, title in enumerate(log_values):
        means = []
        for agent_idx, agent_type in enumerate(agent_types):
            d = data[:, idx * len(agent_types) + agent_idx]
            means.append(sum(d[-10:]) / 10)
            axs[idx].plot(d, alpha=0.6, label=agent_type)
        axs[idx].legend()
        axs[idx].set_title(f"{title} - {', '.join([f'{a}: {m: .2f}' for a, m in zip(agent_types, means)])}")

    fig.tight_layout()
    plt.savefig(plot_filename)
