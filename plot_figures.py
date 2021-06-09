from math import sqrt

import matplotlib.pylab as plt
import numpy as np
import pickle


structs = ['interval', 'snapshot', 'adjtree', 'tvg', 'networkx']
datasets = ['realitymining', 'wikipedia', 'infectious', 'bikeshare', 'wallposts', 'askubuntu', 'enron']

def plot_memory(structs, datasets):
    memory_results = {}
    for struct in structs:
        for dataset_name in datasets:
            memory_results.setdefault(struct, {})[dataset_name] = pickle.load(open(f'memory_results_{struct}_{dataset_name}.pkl', 'rb'))
    fig, ax = plt.subplots()
    width = 0.15
    offset = {'interval': 2 * width, 'snapshot': 1 * width, 'adjtree': 0 * width, 'tvg': -1 * width,'networkx': -2 * width}
    color = {'interval': (0.65, 0.3, 0.3), 'snapshot': (0, 0.75, 0), 'networkx': (0.8, 0.8, 0.8), 'adjtree': (0, 0, .75), 'tvg': (1, 1, 0)}
    labels = {'interval': 'IntervalGraph', 'snapshot': 'SnapshotGraph', 'networkx': 'NetworkX', 'adjtree': 'AdjTree', 'tvg': 'TVG'}
    label_location = np.arange(len(datasets))

    for struct in structs:
        nums = [memory_results[struct][d] for d in memory_results[struct]]
        ax.barh(label_location + offset[struct], nums, width, label=labels[struct], color=color[struct])

    # ax.set_title('Memory')
    ax.set_yticks(label_location)
    ax.set_yticklabels(datasets)
    ax.set_xscale('log')
    ax.set_xlabel('Memory (MB)')
    fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.legend()
    fig.savefig('memory.eps', format='eps')
    plt.show()


def plot_slices(structs, datasets, percent=None):
    if percent is None:
        percents = [1, 5, 10, 20]
    elif isinstance(percent, list):
        percents = percent
    else:
        percents = [percent]

    slice_results = {}
    for struct in structs:
        for dataset_name in datasets:
            slice_results.setdefault(struct, {})[dataset_name] = pickle.load(open(f'slice_results_{struct}_{dataset_name}.pkl', 'rb'))

    avg_results = {}
    for struct in slice_results:
        for dataset_name in slice_results[struct]:
            for percent in slice_results[struct][dataset_name]:
                data = slice_results[struct][dataset_name][percent]
                avg_results.setdefault(struct, {}).setdefault(dataset_name, {})[percent] = sum(data) / len (data)

    fig, ax = plt.subplots()

    label_location = np.arange(len(datasets))
    width = 0.15
    offset = {'interval': 2 * width, 'snapshot': 1 * width, 'adjtree': 0 * width, 'tvg': -1 * width,'networkx': -2 * width}
    color = {'interval': (0.65, 0.3, 0.3), 'snapshot': (0, 0.75, 0), 'networkx': (0.8, 0.8, 0.8), 'adjtree': (0, 0, .75), 'tvg': (1, 1, 0)}
    if len(percents) == 1:
        labels = {'interval': 'IntervalGraph', 'snapshot': 'SnapshotGraph', 'networkx': 'NetworkX', 'adjtree': 'AdjTree', 'tvg': 'TVG'}
    else:
        labels = {'interval': '_IntervalGraph', 'snapshot': '_SnapshotGraph', 'networkx': 'NetworkX', 'adjtree': '_AdjTree', 'tvg': '_TVG'}


    for struct in structs:
        for percent in percents:
            if percents.index(percent) == 0:
                bottom = [0]*len(datasets)
                nums = [avg_results[struct][d][percent] for d in avg_results[struct]]
            else:
                bottom = [avg_results[struct][d][percents[percents.index(percent) - 1]] for d in avg_results[struct]]
                nums = [avg_results[struct][d][percent] - avg_results[struct][d][percents[percents.index(percent) - 1]] for d in avg_results[struct]]
            r, b, g = color[struct]
            if len(percents) != 1:
                p = (percents.index(percent) + 1) * 0.25
                r = 1 - p * (1 - r)
                b = 1 - p * (1 - b)
                g = 1 - p * (1 - g)

            ax.barh(label_location + offset[struct], nums, width, label=labels[struct], left=bottom, color=(r, b, g))


    # ax.set_title('Slice Times by Interval Percent')
    ax.set_yticks(label_location)
    ax.set_yticklabels(datasets)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    if len(percents) == 1:
        ax.set_xlim([None, 1000])
        plt.legend()
        fig.savefig(f'intervalslices_{percents[0]}.eps', format='eps')
    else:
        plt.legend(['1%', '5%', '10%', '20%'])
        fig.savefig(f'intervalslices.eps', format='eps')
    plt.show()


def plot_creation(structs, datasets):
    creation_results = {}
    for struct in structs:
        for dataset_name in datasets:
            creation_results.setdefault(struct, {})[dataset_name] = pickle.load(open(f'creation_time_{struct}_{dataset_name}.pkl', 'rb'))

    fig, ax = plt.subplots()
    width = 0.15
    offset = {'interval': 2 * width, 'snapshot': 1 * width, 'adjtree': 0 * width, 'tvg': -1 * width,'networkx': -2 * width}
    color = {'interval': (0.65, 0.3, 0.3), 'snapshot': (0, 0.75, 0), 'networkx': (0.8, 0.8, 0.8), 'adjtree': (0, 0, .75), 'tvg': (1, 1, 0)}
    labels = {'interval': 'IntervalGraph', 'snapshot': 'SnapshotGraph', 'networkx': 'NetworkX', 'adjtree': 'AdjTree', 'tvg': 'TVG'}
    label_location = np.arange(len(datasets))

    for struct in structs:
        nums = [creation_results[struct][d] for d in creation_results[struct]]
        ax.barh(label_location + offset[struct], nums, width, label=labels[struct], color=color[struct])

    # ax.set_title('Creation Times')
    ax.set_yticks(label_location)
    ax.set_yticklabels(datasets)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.legend()
    fig.savefig('creation.eps', format='eps')
    plt.show()


def plot_compound(datasets):
    score_results = {}
    for dataset_name in datasets:
        score_results[dataset_name] = pickle.load(open(f'score_results_{dataset_name}_log.pkl', 'rb'))
    feature_results = {}
    for dataset_name in datasets:
        feature_results[dataset_name] = pickle.load(open(f'feature_results_interval_{dataset_name}.pkl', 'rb'))
    time_results = {}
    for dataset_name in datasets:
        for result in feature_results[dataset_name]:
            try:
                i = score_results[dataset_name]['X_test'].index(result[:4])
            except:
                continue
            time_results.setdefault('max', {})[dataset_name] = time_results.setdefault('max', {}).setdefault(dataset_name, 0) + max(result[4], result[5])
            time_results.setdefault('tree', {})[dataset_name] = time_results.setdefault('tree', {}).setdefault(dataset_name, 0) + result[5]
            time_results.setdefault('node', {})[dataset_name] = time_results.setdefault('node', {}).setdefault(dataset_name, 0) + result[4]
            if score_results[dataset_name]['Predictions'][i] == 0:
                time_results.setdefault('predict', {})[dataset_name] = time_results.setdefault('predict', {}).setdefault(dataset_name, 0) + result[4]
            else:
                time_results.setdefault('predict', {})[dataset_name] = time_results.setdefault('predict', {}).setdefault(dataset_name, 0) + result[5]
            time_results.setdefault('min', {})[dataset_name] = time_results.setdefault('min', {}).setdefault(dataset_name, 0) + min(result[4], result[5])


    fig, ax = plt.subplots()
    width = 0.15
    offset = {'max': 2 * width, 'tree': 1 * width, 'node': 0 * width, 'predict': -1 * width, 'min': -2 * width}
    color = {'max': (0.65, 0.3, 0.3), 'tree': (0, 0.75, 0), 'min': (0.8, 0.8, 0.8), 'node': (0, 0, .75), 'predict': (1, 1, 0)}
    labels = {'max': 'Maximum Time', 'min': 'Minimum Time', 'node': 'Node Only', 'tree': 'Tree Only', 'predict': 'Prediction'}
    label_location = np.arange(len(datasets))

    for time in time_results:
        nums = [time_results[time][d] for d in time_results[time]]
        ax.barh(label_location + offset[time], nums, width, label=labels[time], color=color[time])

    # ax.set_title('Compound Times')
    ax.set_yticks(label_location)
    ax.set_yticklabels(datasets)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.legend()
    fig.savefig('compound.eps', format='eps')
    plt.show()


def plot_predict(dataset_name):
    feature_results = pickle.load(open(f'feature_results_interval_{dataset_name}.pkl', 'rb'))
    score_results = pickle.load(open(f'score_results_{dataset_name}_log.pkl', 'rb'))

    color = {'Interval': (0.75, 0.3, 0.3), 'Node': (0, 0, .75)}
    nodes = [result[4] for result in feature_results]
    intervals = [result[5] for result in feature_results]

    x = [result[0] for result in feature_results]
    y = [result[1] for result in feature_results]

    n_s = []
    i_s = []
    nx = []
    ny = []
    ix = []
    iy = []

    for i in range(len(intervals)):
        if intervals[i] > nodes[i]:
            n_s.append(5*sqrt(intervals[i] / nodes[i]))
            nx.append(x[i])
            ny.append(y[i])
        else:
            i_s.append(5*sqrt(nodes[i] / intervals[i]))
            ix.append(x[i])
            iy.append(y[i])

    node_color = [color['Node']] * len(n_s)
    interval_color = [color['Interval']] * len(i_s)

    fig, ax = plt.subplots()
    ax.scatter(nx, ny, s=n_s, color=node_color, label='Node', alpha=0.3)
    ax.scatter(ix, iy, s=i_s, color=interval_color, label='Interval', alpha=0.3)

    x = np.linspace(0.1, 50, 100)
    y = (score_results['Coef'][0][0] / score_results['Coef'][0][1])*x + score_results['Intercept']

    ax.plot(x, y, color=(0, 0.75, 0), linewidth=5, label='Model')
    fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.ylim([0, 57])
    ax.set_xlabel('percentOfNodes', fontsize=12)
    ax.set_ylabel('percentOfInterval', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    lgd = plt.legend(fontsize=12, ncol=3)
    lgd.legendHandles[1]._sizes = [100]
    lgd.legendHandles[2]._sizes = [100]
    fig.savefig(f'compoundqueries_{dataset_name}.eps', format='eps')
    plt.show()


def plot_casestudy():
    casestudy = {'networkx': {'Creation': 0.4029717679368332, 'Query': 25.128054951084778, 'Analysis': 74.29869002802297},
                 'intervaltree': {'Creation': 2.20763579395134, 'Query': 2.564769828110002, 'Analysis': 74.29869002802297},
                 'segmenttree': {'Creation': 4.137626645970158, 'Query': 2.5231343526393175, 'Analysis': 74.29869002802297}}
    color = {'Creation': (0.75, 0.3, 0.3), 'Slice': (0, 0.75, 0), 'Analysis': (0, 0, .75)}
    label_location = np.arange(len(casestudy))
    width = 0.2

    fig, ax = plt.subplots()

    bottoms = [0, 0, 0]
    tops = [0, 0, 0]
    for category in casestudy['networkx']:
        i = 0
        for struct in casestudy:
            tops[i] = casestudy[struct][category]
            if category == 'Query':
                bottoms = [0.4029717679368332, 2.20763579395134, 4.137626645970158]
            if category == 'Analysis':
                bottoms = [0.4029717679368332+25.128054951084778, 2.20763579395134+2.564769828110002, 4.137626645970158+2.5231343526393175]
            i += 1

        ax.barh(label_location, tops, width, left=bottoms, color=color[category], label=category)


    ax.set_yticks(label_location)
    ax.set_yticklabels(casestudy.keys())
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.legend()
    fig.savefig('casestudy.eps', format='eps')
    plt.show()


# def plot_error(dataset_name):
#     feature_results = pickle.load(open(f'feature_results_interval_{dataset_name}.pkl', 'rb'))
#     score_results = pickle.load(open(f'score_results_{dataset_name}.pkl', 'rb'))
#
#     x = np.linspace(0.1, 50, 100)
#     y = -(score_results['Coef'][0]*x + score_results['Intercept']) / score_results['Coef'][1]
#
#     delta_times = [result[5] - result[4] for result in feature_results]
#     predicted_times = [score_results['Coef'][0]*result[0] + score_results['Coef'][1]*result[1] + score_results['Intercept'] for result in feature_results]
#
#     x = np.linspace(-0.1, 0.5, 20)
#     fig, ax = plt.subplots()
#     ax.scatter(predicted_times, delta_times, color=(0.75, 0.3, 0.3))
#     ax.plot(x, x, color=(0, 0, .75), linewidth=5)
#
#     ax.set_xlabel('Predicted Delta Time (s)')
#     ax.set_ylabel('Actual Delta Time (s)')
#     fig.tight_layout()
#     fig.set_figheight(5)
#     fig.set_figwidth(10)
#     fig.savefig('error.eps', format='eps')
#     plt.show()
#     plt.show()

# percents can be 1, 5, 10, or 20

## SEPARATE CREATION RESULTS FROM CREATION TIMES

# for struct in structs:
#     for dataset_name in datasets:
#         G, t = pickle.load(open(f'creation_results_{struct}_{dataset_name}.pkl', 'rb'))
#         pickle.dump(G, open(f'structure_{struct}_{dataset_name}.pkl', 'wb'))
#         pickle.dump(t, open(f'creation_time_{struct}_{dataset_name}.pkl', 'wb'))
#         print(f'{struct}_{dataset_name} Done!')

#plot_memory(structs, datasets)
#plot_creation(structs, datasets)
#plot_slices(structs, datasets)
#plot_slices(structs, datasets, 1)
# plot_compound(datasets)
# plot_predict('enron')
# plot_casestudy()
