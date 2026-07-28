import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_cum_max(array):
    return np.maximum.accumulate(np.asarray(array).reshape(-1))


def load_data(alg, task, dim, seed, n_init, indicator, batch_size, max_evals, buffer_size=None, lamb=None):
    path = f'{BASE_DIR}/results/{alg.lower()}/'
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
    ('cibo', 'Ackley',       200, list(range(12)), 200, 'false', 100, 10000, 2000, 10.0),
    ('cibo', 'Rastrigin',    200, list(range(12)), 200, 'false', 100, 10000, 2000, 10.0),
    ('cibo', 'Rosenbrock',   200, list(range(12)), 200, 'false', 100, 10000, 2000, 10.0),
    ('cibo', 'Ackley',       200, list(range(12)), 200, 'true',  100, 10000, 2000, 10.0),
    ('cibo', 'Rastrigin',    200, list(range(12)), 200, 'true',  100, 10000, 2000, 10.0),
    ('cibo', 'Rosenbrock',   200, list(range(12)), 200, 'true',  100, 10000, 2000, 10.0),
])

cmaes_data = load_data_config([
    ('cmaes', 'Ackley',       200, list(range(12)), 200, 'false', 100, 10000, 10000, 10.0),
    ('cmaes', 'Rastrigin',    200, list(range(12)), 200, 'false', 100, 10000, 10000, 10.0),
    ('cmaes', 'Rosenbrock',   200, list(range(12)), 200, 'false', 100, 10000, 10000, 10.0),
    ('cmaes', 'Ackley',       200, list(range(12)), 200, 'true',  100, 10000, 10000, 10.0),
    ('cmaes', 'Rastrigin',    200, list(range(12)), 200, 'true',  100, 10000, 10000, 10.0),
    ('cmaes', 'Rosenbrock',   200, list(range(12)), 200, 'true',  100, 10000, 10000, 10.0),
])

scbo_data = load_data_config([
    ('scbo', 'Ackley',       200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('scbo', 'Rastrigin',    200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('scbo', 'Rosenbrock',   200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('scbo', 'Ackley',       200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('scbo', 'Rastrigin',    200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('scbo', 'Rosenbrock',   200, list(range(12)), 200, 'true',  100, 10000, None, None),
])

pcagp_scbo_data = load_data_config([
    ('pcagp_scbo', 'Ackley',       200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('pcagp_scbo', 'Rastrigin',    200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('pcagp_scbo', 'Rosenbrock',   200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('pcagp_scbo', 'Ackley',       200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('pcagp_scbo', 'Rastrigin',    200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('pcagp_scbo', 'Rosenbrock',   200, list(range(12)), 200, 'true',  100, 10000, None, None),
])
pcagp_scbo_data['Ackley_true']  = pcagp_scbo_data['Ackley_true'][:, :10000]
pcagp_scbo_data['Ackley_false'] = pcagp_scbo_data['Ackley_false'][:, :10000]

cEI_data = load_data_config([
    ('cEI', 'Ackley',       200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('cEI', 'Rastrigin',    200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('cEI', 'Rosenbrock',   200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('cEI', 'Ackley',       200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('cEI', 'Rastrigin',    200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('cEI', 'Rosenbrock',   200, list(range(12)), 200, 'true',  100, 10000, None, None),
])

logcEI_data = load_data_config([
    ('logcEI', 'Ackley',       200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('logcEI', 'Rastrigin',    200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('logcEI', 'Rosenbrock',   200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('logcEI', 'Ackley',       200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('logcEI', 'Rastrigin',    200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('logcEI', 'Rosenbrock',   200, list(range(12)), 200, 'true',  100, 10000, None, None),
])

diffopt_data = load_data_config([
    ('diffopt', 'Ackley',       200, list(range(12)), 200, 'false', 100, 10000, 1000, 10.0),
    ('diffopt', 'Rastrigin',    200, list(range(12)), 200, 'false', 100, 10000, 1000, 10.0),
    ('diffopt', 'Rosenbrock',   200, list(range(12)), 200, 'false', 100, 10000, 1000, 10.0),
    ('diffopt', 'Ackley',       200, list(range(12)), 200, 'true',  100, 10000, 1000, 10.0),
    ('diffopt', 'Rastrigin',    200, list(range(12)), 200, 'true',  100, 10000, 1000, 10.0),
    ('diffopt', 'Rosenbrock',   200, list(range(12)), 200, 'true',  100, 10000, 1000, 10.0),
])

cobyla_data = load_data_config([
    ('cobyla', 'Ackley',     200, list(range(12)), 200, 'false', 100, 10000, 10000, 1.0),
    ('cobyla', 'Rastrigin',  200, list(range(12)), 200, 'false', 100, 10000, 10000, 1.0),
    ('cobyla', 'Rosenbrock', 200, list(range(12)), 200, 'false', 100, 10000, 10000, 1.0),
    ('cobyla', 'Ackley',     200, list(range(12)), 200, 'true',  100, 10000, 10000, 1.0),
    ('cobyla', 'Rastrigin',  200, list(range(12)), 200, 'true',  100, 10000, 10000, 1.0),
    ('cobyla', 'Rosenbrock', 200, list(range(12)), 200, 'true',  100, 10000, 10000, 1.0),
])

sobol_data = load_data_config([
    ('sobol', 'Ackley',     200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('sobol', 'Rastrigin',  200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('sobol', 'Rosenbrock', 200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('sobol', 'Ackley',     200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('sobol', 'Rastrigin',  200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('sobol', 'Rosenbrock', 200, list(range(12)), 200, 'true',  100, 10000, None, None),
])

furbo_data = load_data_config([
    ('furbo', 'Ackley',     200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('furbo', 'Rastrigin',  200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('furbo', 'Rosenbrock', 200, list(range(12)), 200, 'false', 100, 10000, None, None),
    ('furbo', 'Ackley',     200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('furbo', 'Rastrigin',  200, list(range(12)), 200, 'true',  100, 10000, None, None),
    ('furbo', 'Rosenbrock', 200, list(range(12)), 200, 'true',  100, 10000, None, None),
])


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_paper_results(algs_list, data_list, color_list=None):
    synthetic_false = ['Rastrigin_false', 'Ackley_false', 'Rosenbrock_false']
    synthetic_true  = ['Rastrigin_true',  'Ackley_true',  'Rosenbrock_true']

    if color_list is None:
        color_list = plt.cm.tab10.colors[:len(algs_list)]
    color_map = dict(zip(algs_list, color_list))

    fig_syn, axes_syn = plt.subplots(2, 3, figsize=(10, 7))

    for row, tasks in enumerate([synthetic_false, synthetic_true]):
        for col, task in enumerate(tasks):
            ax = axes_syn[row, col]

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

            finite_after_cutoff = []
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
                        tail = mean_data[100:]
                        finite_tail = tail[np.isfinite(tail)]
                        if finite_tail.size > 0:
                            finite_after_cutoff.append(np.max(finite_tail))
                except Exception as e:
                    print(f"Error {alg} {task}: {e}")

            title = task.replace('_false', '').replace('_true', '')
            suffix = '-200D (Indicator)' if row == 1 else '-200D'
            ax.set_title(f"{title}{suffix}", fontsize=18, pad=10)
            ax.set_xlabel("Number of Evaluations\n", fontsize=14)
            ax.set_ylabel("Best Value Found (log)" if 'Rosenbrock' in task else "Best Value Found", fontsize=14)
            ax.grid(True)
            if 'Rosenbrock' in task:
                ax.set_yscale('log')
            if task == 'Rastrigin_true':
                ax.set_ylim(top=2800)
            if task == 'Ackley_true':
                ax.set_ylim(top=9)

    # Collect from both rows so a method that only appears in one row still shows up.
    label_to_handle = {}
    for ax in (axes_syn[1, 0], axes_syn[0, 0], axes_syn[0, 1], axes_syn[0, 2]):
        handles, labels = ax.get_legend_handles_labels()
        for label, handle in zip(labels, handles):
            label_to_handle.setdefault(label, handle)
    legend_order = ['CiBO (Ours)', 'cEI', 'DiffOPT', 'LogcEI', 'SCBO',
                    'CMA-ES', 'PCAGP-SCBO', 'COBYLA', 'FuRBO', 'Sobol']
    ordered_handles = [label_to_handle[l] for l in legend_order if l in label_to_handle]
    ordered_labels  = [l for l in legend_order if l in label_to_handle]
    leg = fig_syn.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                         ncol=5, fontsize=18)
    for text in leg.get_texts():
        if 'CiBO' in text.get_text():
            text.set_fontweight('bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    out_path = f"{BASE_DIR}/figures/synthetic.pdf"
    fig_syn.savefig(out_path, bbox_inches='tight')
    print(f"Saved → {out_path}")
    plt.close(fig_syn)


algs_list  = ['CiBO (Ours)', 'DiffOPT', 'SCBO', 'PCAGP-SCBO', 'cEI', 'LogcEI', 'CMA-ES', 'COBYLA', 'FuRBO', 'Sobol']
data_list  = [cibo_data, diffopt_data,
              scbo_data, pcagp_scbo_data, cEI_data, logcEI_data, cmaes_data, cobyla_data, furbo_data, sobol_data]
color_list = ['#1f77b4', '#cc6600', '#b5599e',
              '#6d3f34', '#8b8b00', '#5f5f5f', '#724b96', '#1e7a1e', '#a31d1d', '#0e8e9e']

plot_paper_results(algs_list, data_list, color_list=color_list)
