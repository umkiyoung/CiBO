import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_cum_max(array):
    return np.maximum.accumulate(np.asarray(array).reshape(-1))


def load_data(alg, task, dim, seed, n_init, indicator, batch_size, max_evals, buffer_size=None, lamb=None):
    path = f'{BASE_DIR}/results/{alg}/'
    if lamb is None and buffer_size is not None:
        filename = f'{alg}_{task}_{dim}_{seed}_{n_init}_{indicator}_{batch_size}_{buffer_size}_{max_evals}_{max_evals}.npy'
    elif buffer_size is None and lamb is None:
        filename = f'{alg}_{task}_{dim}_{seed}_{n_init}_{indicator}_{batch_size}_{max_evals}_{max_evals}.npy'
    else:
        filename = f'{alg}_{task}_{dim}_{seed}_{n_init}_{indicator}_{lamb}_{batch_size}_{buffer_size}_{max_evals}_{max_evals}.npy'
    try:
        data = np.load(f"{path}{filename}")
    except FileNotFoundError:
        print(f"File not found: {path}{filename}")
        return None
    return get_cum_max(data)


def load_data_all(alg, task, dim, seed_list, n_init, indicator, batch_size, max_evals, buffer_size=None, lamb=None):
    data_all = []
    for seed in seed_list:
        data = load_data(alg, task, dim, seed, n_init, indicator, batch_size, max_evals, buffer_size, lamb)
        if data is not None:
            data_all.append(data)
    if len(data_all) == 0:
        return None
    return np.array(data_all)


def load_data_config(configs):
    data_all = {}
    for config in configs:
        alg, task, dim, seed_list, n_init, indicator, batch_size, max_evals, buffer_size, lamb = config
        print(f"Loading {alg} {task} {indicator}")
        data = load_data_all(alg, task, dim, seed_list, n_init, indicator, batch_size, max_evals, buffer_size, lamb)
        data_all[f"{task}_{indicator}"] = data
    return data_all


# ── Load data ────────────────────────────────────────────────────────────────

cibo_data = load_data_config([
    ('cibo', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, 1000,  5.0),
    ('cibo', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, 1000,  5.0),
    ('cibo', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, 1000, 10.0),
    ('cibo', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, 1000,  5.0),
])

cmaes_data = load_data_config([
    ('cmaes', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, 2000, 3.0),
    ('cmaes', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, 2000, 3.0),
    ('cmaes', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, 2000, 3.0),
    ('cmaes', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, 2000, 3.0),
])

scbo_data = load_data_config([
    ('scbo', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('scbo', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, None, None),
    ('scbo', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('scbo', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
])

pcagp_scbo_data = load_data_config([
    ('pcagp_scbo', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('pcagp_scbo', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, None, None),
    ('pcagp_scbo', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('pcagp_scbo', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
])

cEI_data = load_data_config([
    ('cEI', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('cEI', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, None, None),
    ('cEI', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('cEI', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
])

logcEI_data = load_data_config([
    ('logcEI', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('logcEI', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, None, None),
    ('logcEI', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('logcEI', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
])

diffopt_data = load_data_config([
    ('diffopt', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, 500, 3.0),
    ('diffopt', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, 500, 3.0),
    ('diffopt', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, 500, 3.0),
    ('diffopt', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, 500, 3.0),
])

cobyla_data = load_data_config([
    ('cobyla', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, 2000, 1.0),
    ('cobyla', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, 2000, 1.0),
    ('cobyla', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, 2000, 1.0),
    ('cobyla', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, 2000, 1.0),
])

furbo_data = load_data_config([
    ('furbo', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('furbo', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('furbo', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, None, None),
    ('furbo', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
])

sobol_data = load_data_config([
    ('sobol', 'RoverPlanning', 60,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('sobol', 'HalfCheetah',  102,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
    ('sobol', 'Mopta',        124,  [0,1,2,3], 200, 'false', 20, 2000, None, None),
    ('sobol', 'DNA',          180,  [0,1,2,3], 200, 'false', 50, 2000, None, None),
])


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_realworld_results(algs_list, data_list, color_list=None):
    real_tasks = [
        ('RoverPlanning_false', 'RoverPlanning-60D'),
        ('HalfCheetah_false',   'HalfCheetah-102D'),
        ('Mopta_false',         'Mopta-124D'),
        ('DNA_false',           'DNA-180D'),
    ]

    if color_list is None:
        color_list = plt.cm.tab10.colors[:len(algs_list)]
    color_map = dict(zip(algs_list, color_list))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))

    for idx, (task, title) in enumerate(real_tasks):
        ax = axes[idx]

        finite_max_values = []
        for alg, data in zip(algs_list, data_list):
            if task in data and data[task] is not None:
                try:
                    mean_data = -np.mean(data[task], axis=0).reshape(-1)
                    finite = mean_data[np.isfinite(mean_data)]
                    if finite.size > 0:
                        finite_max_values.append(np.max(finite))
                except Exception:
                    continue

        finite_global_max = max(finite_max_values) if finite_max_values else 1
        inf_line_position = finite_global_max * 1.1

        for alg, data in zip(algs_list, data_list):
            try:
                if task not in data or data[task] is None:
                    continue
                mean_data = -np.mean(data[task], axis=0).reshape(-1)
                std_data  =  np.std(data[task],  axis=0).reshape(-1)

                is_ours = (color_map[alg] == '#1f77b4')
                lw     = 3.5 if is_ours else 2.5
                zorder = 10  if is_ours else None

                if np.all(~np.isfinite(mean_data)):
                    ax.plot([inf_line_position] * len(mean_data),
                            color=color_map[alg], linestyle='--', label=alg,
                            linewidth=lw, zorder=zorder)
                else:
                    mean_clean = np.copy(mean_data)
                    mean_clean[~np.isfinite(mean_clean)] = inf_line_position
                    ax.plot(mean_clean, color=color_map[alg], label=alg,
                            linewidth=lw, zorder=zorder)
                    ax.fill_between(range(len(mean_clean)),
                                    mean_clean - std_data,
                                    mean_clean + std_data,
                                    alpha=0.1, color=color_map[alg])
            except Exception as e:
                print(f"Error {alg} {task}: {e}")

        ax.set_title(title, fontsize=18, pad=10)
        ax.set_xlabel("Number of Evaluations", fontsize=14)
        ax.set_ylabel("Best Value Found", fontsize=14)
        ax.grid(True)
        if task == 'RoverPlanning_false':
            ax.set_ylim(top=15)
        if task == 'HalfCheetah_false':
            ax.set_ylim(bottom=-3200, top=500)
        if task == 'DNA_false':
            ax.set_ylim(bottom=0.290)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = ['CiBO (Ours)', 'cEI', 'DiffOPT', 'LogcEI', 'SCBO',
                    'CMA-ES', 'PCAGP-SCBO', 'COBYLA', 'FuRBO', 'Sobol']
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[l] for l in legend_order if l in label_to_handle]
    ordered_labels  = [l for l in legend_order if l in label_to_handle]
    leg = fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
                     ncol=5, fontsize=20)
    for text in leg.get_texts():
        if 'CiBO' in text.get_text():
            text.set_fontweight('bold')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.32, bottom=0.12)

    out_path = f"{BASE_DIR}/figures/realworld.pdf"
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig)


algs_list  = ['CiBO (Ours)', 'DiffOPT', 'SCBO', 'PCAGP-SCBO', 'cEI', 'LogcEI', 'CMA-ES', 'COBYLA', 'FuRBO', 'Sobol']
data_list  = [cibo_data, diffopt_data,
              scbo_data, pcagp_scbo_data, cEI_data, logcEI_data, cmaes_data, cobyla_data, furbo_data, sobol_data]
color_list = ['#1f77b4', '#cc6600', '#b5599e',
              '#6d3f34', '#8b8b00', '#5f5f5f', '#724b96', '#1e7a1e', '#a31d1d', '#0e8e9e']

plot_realworld_results(algs_list, data_list, color_list=color_list)
