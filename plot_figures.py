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

    fig, ax = plt.subplots(1, 2, sharex=True)

    label_location = np.arange(len(datasets))
    width = 0.15
    offset = {'interval': 2 * width, 'snapshot': 1 * width, 'adjtree': 0 * width, 'tvg': -1 * width,'networkx': -2 * width}
    color = {'interval': (0.65, 0.3, 0.3), 'snapshot': (0, 0.75, 0), 'networkx': (0.8, 0.8, 0.8), 'adjtree': (0, 0, .75), 'tvg': (1, 1, 0)}
    if len(percents) == 1:
        labels = {'interval': 'IntervalGraph', 'snapshot': 'SnapshotGraph', 'networkx': 'NetworkX', 'adjtree': 'AdjTree', 'tvg': 'TVG'}
    else:
        labels = {'interval': '_IntervalGraph', 'snapshot': '_SnapshotGraph', 'networkx': 'NetworkX', 'adjtree': '_AdjTree', 'tvg': '_TVG'}
    lines = []
    for struct in structs:
        for i in range(len(percents)):
            percent = percents[i]
            l = ax[i].barh(label_location + offset[struct], [avg_results[struct][d][percent] for d in avg_results[struct]], width, label=labels[struct], color=color[struct])
            if i == 0:
                lines.append(l)

    # ax.set_title('Slice Times by Interval Percent')
    ax[0].set_yticks(label_location)
    ax[0].set_yticklabels(datasets)
    ax[1].set_yticklabels([])
    ax[0].set_xlabel('1% time slice')
    ax[1].set_xlabel('5% time slice')
    plt.xscale('log')
    fig.text(0.5, 0, 'Time (s)', ha='center')

    fig.set_figheight(3)
    fig.set_figwidth(8)

    if len(percents) == 1:
        ax[0].set_xlim([None, 1000])
        ax[1].set_xlim([None, 1000])
        plt.legend(frameon=False, fontsize=8, loc=9,
                   bbox_to_anchor=(0.5, 1.1))
        fig.savefig(f'intervalslices_{percents[0]}.eps', format='eps')
    else:
        fig.legend(handles=lines,
                   labels=['IntervalGraph', 'SnapshotGraph', 'AdjTree', 'TVG', 'NetworkX'],
                   frameon=False, ncol=5, loc=9, bbox_to_anchor=(0.5, 1.15))
        # fig.tight_layout()
        fig.savefig(f'intervalslices_15.eps', format='eps', bbox_inches='tight')

    plt.show()


def plot_creation(structs, datasets):
    creation_results = {}
    for struct in structs:
        for dataset_name in datasets:
            creation_results.setdefault(struct, {})[dataset_name] = pickle.load(open(f'creation_results_{struct}_{dataset_name}.pkl', 'rb'))

    fig, ax = plt.subplots()
    width = 0.15
    offset = {'interval': 2 * width, 'snapshot': 1 * width, 'adjtree': 0 * width, 'tvg': -1 * width,'networkx': -2 * width}
    color = {'interval': (0.65, 0.3, 0.3), 'snapshot': (0, 0.75, 0), 'networkx': (0.8, 0.8, 0.8), 'adjtree': (0, 0, .75), 'tvg': (1, 1, 0)}
    labels = {'interval': 'IntervalGraph', 'snapshot': 'SnapshotGraph', 'networkx': 'NetworkX', 'adjtree': 'AdjTree', 'tvg': 'TVG'}
    label_location = np.arange(len(datasets))

    for struct in structs:
        nums = [creation_results[struct][d][1] for d in creation_results[struct]]
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
    RMSE = []
    R2 = []
    ACC = []
    for dataset_name in datasets:
        score_results[dataset_name] = pickle.load(open(f'score_results_{dataset_name}_lin.pkl', 'rb'))
        RMSE.append((dataset_name, score_results[dataset_name]['RMSE']))
        R2.append((dataset_name, score_results[dataset_name]['R2 Score']))
        ACC.append((dataset_name, score_results[dataset_name]['Accuracy']))
    print('RMSE')
    for v in RMSE:
        print(v)
    print('R2')
    for v in R2:
        print(v)
    print('ACC')
    for v in ACC:
        print(v)

    feature_results = {}
    for dataset_name in datasets:
        feature_results[dataset_name] = pickle.load(open(f'feature_results_interval_{dataset_name}.pkl', 'rb'))
    time_results = {}
    for dataset_name in datasets:
        for result in feature_results[dataset_name]:
            node_time, interval_time = result[1][0], result[1][1]
            time_results.setdefault('max', {})[dataset_name] = time_results.setdefault('max', {}).setdefault(dataset_name, 0) + max(node_time, interval_time)
            time_results.setdefault('tree', {})[dataset_name] = time_results.setdefault('tree', {}).setdefault(dataset_name, 0) + interval_time
            time_results.setdefault('node', {})[dataset_name] = time_results.setdefault('node', {}).setdefault(dataset_name, 0) + node_time
            [[pred_node, pred_interval]] = score_results[dataset_name]['Model'].predict([[result[0][i] for i in [0, 1]]])
            if pred_node < pred_interval:
                time_results.setdefault('predict', {})[dataset_name] = time_results.setdefault('predict', {}).setdefault(dataset_name, 0) + node_time
            else:
                time_results.setdefault('predict', {})[dataset_name] = time_results.setdefault('predict', {}).setdefault(dataset_name, 0) + interval_time
            time_results.setdefault('min', {})[dataset_name] = time_results.setdefault('min', {}).setdefault(dataset_name, 0) + min(node_time, interval_time)

    fig, ax = plt.subplots()
    width = 0.15
    offset = {'max': 2 * width, 'tree': 1 * width, 'node': 0 * width, 'predict': -1 * width, 'min': -2 * width}
    color = {'max': (0.65, 0.3, 0.3), 'tree': (0, 0.75, 0), 'min': (0.8, 0.8, 0.8), 'node': (0, 0, .75), 'predict': (1, 1, 0)}
    labels = {'max': 'Maximum Time', 'min': 'Minimum Time', 'node': 'Node Only', 'tree': 'Tree Only', 'predict': 'Prediction'}
    label_location = np.arange(len(datasets))

    print('Prediction Time')
    for d, t in time_results['predict'].items():
        print(t)
    #
    # print('Node Time')
    # for d, t in time_results['node'].items():
    #     print(t)
    #
    # print('Min Time')
    # for d, t in time_results['min'].items():
    #     print(t)

    for time in time_results:
        nums = [time_results[time][d] for d in time_results[time]]
        ax.barh(label_location + offset[time], nums, width, label=labels[time], color=color[time])

    # ax.set_title('Compound Times')
    ax.set_yticks(label_location)
    ax.set_yticklabels(datasets)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.set_figheight(3.75)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.legend()
    fig.savefig(f'compound.eps', format='eps')
    plt.show()


def plot_compound_compare(datasets):
    lin_score_results = {}
    log_score_results = {}

    for dataset_name in datasets:
        lin_score_results[dataset_name] = pickle.load(open(f'score_results_{dataset_name}_lin.pkl', 'rb'))
        log_score_results[dataset_name] = pickle.load(open(f'score_results_{dataset_name}_log.pkl', 'rb'))

    feature_results = {}
    for dataset_name in datasets:
        feature_results[dataset_name] = pickle.load(open(f'feature_results_interval_{dataset_name}.pkl', 'rb'))
    time_results = {}
    for dataset_name in datasets:
        for result in feature_results[dataset_name]:
            node_time, interval_time = result[1][0], result[1][1]
            time_results.setdefault('max', {})[dataset_name] = time_results.setdefault('max', {}).setdefault(dataset_name, 0) + max(node_time, interval_time)
            time_results.setdefault('tree', {})[dataset_name] = time_results.setdefault('tree', {}).setdefault(dataset_name, 0) + interval_time
            time_results.setdefault('node', {})[dataset_name] = time_results.setdefault('node', {}).setdefault(dataset_name, 0) + node_time

            pred = log_score_results[dataset_name]['Model'].predict([[result[0][i] for i in [0, 1]]])
            if pred == 0:
                time_results.setdefault('log', {})[dataset_name] = time_results.setdefault('log', {}).setdefault(dataset_name, 0) + node_time
            else:
                time_results.setdefault('log', {})[dataset_name] = time_results.setdefault('log', {}).setdefault(dataset_name, 0) + interval_time

            [[pred_node, pred_interval]] = lin_score_results[dataset_name]['Model'].predict([[result[0][i] for i in [0, 1]]])
            if pred_node < pred_interval:
                time_results.setdefault('lin', {})[dataset_name] = time_results.setdefault('lin', {}).setdefault(dataset_name, 0) + node_time
            else:
                time_results.setdefault('lin', {})[dataset_name] = time_results.setdefault('lin', {}).setdefault(dataset_name, 0) + interval_time

            time_results.setdefault('min', {})[dataset_name] = time_results.setdefault('min', {}).setdefault(dataset_name, 0) + min(node_time, interval_time)


    fig, ax = plt.subplots()
    width = 0.15
    offset = {'max': 2.5 * width, 'tree': 1.5 * width, 'node': 0.5 * width, 'log': -0.5 * width,'lin': -1.5 * width, 'min': -2.5 * width}
    color = {'max': (0.65, 0.3, 0.3), 'tree': (0, 0.75, 0), 'min': (0.8, 0.8, 0.8), 'node': (0, 0, .75), 'lin': (1, 1, 0), 'log': (0.5, 0, 0.5)}
    labels = {'max': 'Maximum Time', 'min': 'Minimum Time', 'node': 'Node Only', 'tree': 'Tree Only',
              'lin': 'Dual Linear Regression', 'log': 'Logistic Regression'}
    label_location = np.arange(len(datasets))

    # print('Prediction Time')
    # for d, t in time_results['predict'].items():
    #     print(t)
    #
    # print('Node Time')
    # for d, t in time_results['node'].items():
    #     print(t)
    #
    # print('Min Time')
    # for d, t in time_results['min'].items():
    #     print(t)

    for time in time_results:
        nums = [time_results[time][d] for d in time_results[time]]
        ax.barh(label_location + offset[time], nums, width, label=labels[time], color=color[time])

    # ax.set_title('Compound Times')
    ax.set_yticks(label_location)
    ax.set_yticklabels(datasets)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.set_figheight(3.75)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.legend()
    fig.savefig(f'compound.eps', format='eps')
    plt.show()


def plot_predict(dataset_name):
    feature_results = pickle.load(open(f'feature_results_interval_{dataset_name}.pkl', 'rb'))
    score_results = pickle.load(open(f'score_results_{dataset_name}_lin.pkl', 'rb'))

    color = {'Interval': (0.75, 0.3, 0.3), 'Node': (0, 0, .75)}
    nodeX, nodeY, intervalX, intervalY, nodeS, intervalS = [], [], [], [], [], []

    for t in feature_results[:2500]:
        nodePercent, intervalPercent, nodeTime, intervalTime = t[0][0], t[0][1], t[1][0], t[1][1]
        if nodeTime < intervalTime:
            nodeX.append(nodePercent*100)
            nodeY.append(intervalPercent*100)
            nodeS.append(5*sqrt(intervalTime / nodeTime))
        else:
            intervalX.append(nodePercent*100)
            intervalY.append(intervalPercent*100)
            intervalS.append(5*sqrt(nodeTime / intervalTime))

    nodeC = [color['Node']] * len(nodeS)
    intervalC = [color['Interval']] * len(intervalS)

    fig, ax = plt.subplots()
    ax.scatter(nodeX, nodeY, s=nodeS, color=nodeC, label='Node', alpha=0.3)
    ax.scatter(intervalX, intervalY, s=intervalS, color=intervalC, label='Interval', alpha=0.3)

    x = np.linspace(0.1, 45, 100)
    coef, inter = score_results['Coef'], score_results['Intercept']
    y = ((coef[0][0] + coef[1][0])*x + inter[0] + inter[1])/(coef[0][1] + coef[1][1])

    ax.plot(x, y, color=(0, 0.75, 0), linewidth=5, label='Model')
    # fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    # plt.tight_layout(pad=0.5)
    plt.ylim([0, 58])
    ax.set_xlabel('Node Percent', fontsize=12)
    ax.set_ylabel('Interval Percent', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    lgd = plt.legend(fontsize=12, ncol=3)
    lgd.legendHandles[1]._sizes = [100]
    lgd.legendHandles[2]._sizes = [100]
    fig.savefig(f'compoundqueries_{dataset_name}_2500.eps', format='eps')
    plt.show()

    print('Coef', score_results['Coef'])
    print('Intercept', score_results['Intercept'])


def plot_predict_compare(dataset_name):
    feature_results = pickle.load(open(f'feature_results_interval_{dataset_name}.pkl', 'rb'))
    score_results = pickle.load(open(f'score_results_{dataset_name}_lin.pkl', 'rb'))
    color = {'Points': (0.75, 0.3, 0.3), 'Line': (0, 0, .75)}

    model = score_results['Model']

    X = [result[0] for result in feature_results]
    y = [result[1] for result in feature_results]
    y_pred = model.predict(X)

    fig, axes = plt.subplots(1, 2)

    maxX = max([i[0] for i in y_pred])*1.1
    maxY = max([i[1] for i in y_pred])*1.1
    axes[0].scatter([i[0] for i in y], [i[0] for i in y_pred], color=color['Points'])
    axes[1].scatter([i[1] for i in y], [i[1] for i in y_pred], color=color['Points'])

    x = np.linspace(0, maxX, 100)
    axes[0].plot(x, x, color=color['Line'])
    x = np.linspace(0, maxY, 100)
    axes[1].plot(x, x, color=color['Line'])

    fig.set_figheight(4)
    fig.set_figwidth(8)

    axes[0].set_title('Node Time')
    axes[1].set_title('Interval Time')
    axes[0].tick_params(axis='both', labelsize=12)
    axes[1].tick_params(axis='both', labelsize=12)
    fig.text(0.5, 0.02, 'Actual', ha='center', va='center', fontsize=12)
    fig.text(0.02, 0.5, 'Predicted', ha='center', va='center', rotation='vertical', fontsize=12)

    axes[0].set_xlim([0, maxX])
    axes[0].set_ylim([0, maxX])
    axes[1].set_xlim([0, maxY])
    axes[1].set_ylim([0, maxY])
    axes[1].set_xlabel(' ', fontsize=12)
    axes[0].set_ylabel(' ', fontsize=12)

    # plt.subplots_adjust(wspace=0.)
    fig.savefig(f'comparison_{dataset_name}.eps', format='eps')
    plt.show()


def plot_train_time(datasets):
    creation_results = {}
    for dataset_name in datasets:
        creation_results[dataset_name] = pickle.load(open(f'creation_results_interval_{dataset_name}.pkl', 'rb'))

    fig, ax = plt.subplots()
    color = [(0, 0, 0.75), (0.75, 0.3, 0.3)]

    print(creation_results)
    labels = list(creation_results.keys())
    ax.barh(labels, [creation_results[r][1] for r in creation_results], color=color[0], label='Creation')
    ax.barh(labels, [creation_results[r][2] for r in creation_results], color=color[1],
            left=[creation_results[r][1] for r in creation_results], label='Model')
    # ax.set_title('Creation Times')
    ax.set_yticklabels(datasets)
    ax.set_xscale('log')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.xlim([1, 600])
    plt.legend()
    fig.savefig('model_creation_log.eps', format='eps')
    plt.show()


def plot_accuracy(datasets):
    creation_results = {}
    score_results = {}
    percents = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for dataset_name in datasets:
        for percent in range(1, 11):
            creation_results.setdefault(dataset_name, {})[percent] = pickle.load(open(f'creation_results_interval_{dataset_name}_{percents[percent-1]}.pkl', 'rb'))[2]
            score_results.setdefault(dataset_name, {})[percent] = pickle.load(open(f'score_results_{dataset_name}_lin_{percent}.pkl', 'rb'))['Accuracy']

    fig, axes = plt.subplots(1, 2)
    color = {'enron': (0.65, 0.3, 0.3), 'askubuntu': (0, 0.75, 0),
             'wallposts': (0.8, 0.8, 0.8), 'bikeshare': (0, 0, .75),
             'infectious': (1, 1, 0), 'wikipedia': (0.7, 0.5, 0),
             'realitymining': (0.5, 0, 0.5)}

    fig.set_figheight(4)
    fig.set_figwidth(8)

    for dataset_name in score_results:
        axes[0].plot(percents, [score_results[dataset_name][r] for r in score_results[dataset_name]], label=dataset_name, color=color[dataset_name])
        axes[1].plot(percents, [creation_results[dataset_name][r] for r in creation_results[dataset_name]], label=dataset_name, color=color[dataset_name])

    axes[0].set_xlabel(' ', fontsize=12)
    axes[1].set_yscale('log')
    axes[0].set_ylabel('Time (s)')

    fig.text(0.5, 0.02, 'Training Size', ha='center', va='center', fontsize=12)
    # axes[0].set_ylabel(' ', fontsize=12)
    plt.legend()
    fig.savefig('accuracy.eps', format='eps')
    plt.show()


def plot_casestudy():
    interval_creation, interval_slice, networkx_creation, networkx_slice, raw_slice, analysis = pickle.load(open(f'case_study_results.pkl', 'rb'))

    label_location = np.arange(2)
    fig, ax = plt.subplots()

    create_nums = [interval_creation, networkx_creation]
    slice_nums = [interval_slice, networkx_slice]
    analysis_nums = [analysis]*2

    ax.barh(label_location, create_nums, 0.5, label='Creation', color=(0.65, 0.3, 0.3))
    ax.barh(label_location, slice_nums, 0.5, label='Slice', left=create_nums, color=(0, 0.75, 0))
    ax.barh(label_location, analysis_nums, 0.5, label='Analysis', left=[c + s for c, s in zip(create_nums, slice_nums)], color=(0, 0, .75))

    ax.set_yticks(label_location)
    ax.set_yticklabels(['IntervalGraph', 'NetworkX'])
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.set_figheight(2)
    fig.set_figwidth(6)
    plt.tight_layout(pad=0.2)
    plt.legend()
    fig.savefig('casestudy.eps', format='eps')
    plt.show()


def plot_feat_times(datasets):
    feature_results = {}
    score_results = {}
    models = ['2', '4', '7']
    for dataset_name in datasets:
        for feats in models:
            features = pickle.load(open(f'feature_results_interval_{dataset_name}.pkl', 'rb'))
            feature_results[dataset_name] = [sum([f[2][:2] for f in features]), sum([f[2][:4] for f in features]), sum([f[2][:7] for f in features])]
            score_results.setdefault(dataset_name, {})[feats] = pickle.load(open(f'score_results_{dataset_name}_lin_{feats}.pkl', 'rb'))['Accuracy']

    color = {'2': (0.65, 0.3, 0.3), '4': (0, 0.75, 0), '7': (0, 0, .75)}
    width = 0.15
    offset = {'2': -width, '4': 0, '7': width}
    labels = {'2': 'Percent Only', '4': '+ Degree/Lifespan', '7': '+ BinFeatures'}
    label_location = np.arange(len(datasets))

    fig, axes = plt.subplots(1, 2)
    fig.set_figheight(4)
    fig.set_figwidth(8)

    for i in range(len(models)):
        m = models[i]
        times = [feature_results[d][i] for d in datasets]
        acc = [score_results[d][m] for d in datasets]
        axes[0].barh(label_location + offset[m], acc, width, label=labels[m], color=color[m])
        axes[1].barh(label_location + offset[m], times, width, label=labels[m], color=color[m])

    axes[0].set_xlim([0.6, 1])
    axes[1].set_xscale('log')
    axes[0].set_yticks(label_location)
    axes[0].set_yticklabels(datasets)

    plt.legend()
    fig.savefig('feat_times.eps', format='eps')
    plt.show()


def _separate_creation_results(dataset_name, percent):
    r = pickle.load(open(f'creation_results_interval_{dataset_name}_{percent}.pkl', 'rb'))
    newR = [None] + list(r[1:])
    pickle.dump(newR, open(f'creation_results_interval_{dataset_name}_{percent}_2.pkl', 'wb'))
    return dataset_name, percent


plot_memory(structs, datasets)
plot_creation(structs, datasets)
plot_slices(structs, datasets)
plot_slices(structs, datasets, [1, 5])
plot_slices(structs, datasets, 5)
plot_compound(datasets)
plot_compound_compare(datasets)
plot_predict('infectious')
plot_predict_compare('infectious')
plot_train_time(datasets)
plot_accuracy(datasets)
plot_feat_times(datasets)
plot_casestudy()