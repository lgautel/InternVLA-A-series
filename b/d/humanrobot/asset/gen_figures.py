"""Generate comparison figures for humanoid vs wheeled VLA analysis."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})


def fig1_action_space_comparison():
    """Bar chart comparing action space dimensionality across robot types."""
    categories = [
        'Wheeled\nBase',
        'Single Arm\n(Franka)',
        'Bimanual\n(ALOHA)',
        'Humanoid\nUpper Body\n(Helix)',
        'Full Humanoid\n(Optimus)',
    ]
    dofs = [2, 7, 14, 35, 78]
    colors = ['#4CAF50', '#2196F3', '#2196F3', '#FF5722', '#FF5722']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, dofs, color=colors, edgecolor='white', linewidth=1.5)

    for bar, dof in zip(bars, dofs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                f'{dof} DoF', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Degrees of Freedom (DoF)')
    ax.set_title('Action Space Dimensionality: Wheeled vs Humanoid Robots')
    ax.set_ylim(0, 95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', label='Wheeled / Mobile'),
        mpatches.Patch(facecolor='#2196F3', label='Arm Manipulation'),
        mpatches.Patch(facecolor='#FF5722', label='Humanoid'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    fig.savefig('fig1_action_space.png')
    plt.close(fig)
    print('Saved fig1_action_space.png')


def fig2_control_frequency():
    """Horizontal bar chart comparing control frequencies."""
    components = [
        'VLA Reasoning\n(Wheeled)',
        'VLA Reasoning\n(Humanoid S2)',
        'Action Expert\n(Humanoid S1)',
        'WBC / Balance\n(Humanoid)',
        'PD Controller\n(Humanoid)',
    ]
    freq_low = [5, 5, 50, 200, 200]
    freq_high = [30, 25, 200, 200, 1000]
    colors = ['#4CAF50', '#FF5722', '#FF5722', '#FF5722', '#FF5722']

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(components))

    for i, (lo, hi) in enumerate(zip(freq_low, freq_high)):
        ax.barh(y_pos[i], hi - lo, left=lo, color=colors[i], alpha=0.8, height=0.6)
        ax.barh(y_pos[i], lo, color=colors[i], alpha=0.4, height=0.6)
        label = f'{lo}–{hi} Hz' if lo != hi else f'{lo} Hz'
        ax.text(hi + 15, y_pos[i], label, va='center', fontweight='bold', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(components)
    ax.set_xlabel('Control Frequency (Hz)')
    ax.set_title('Control Frequency Requirements: Wheeled vs Humanoid')
    ax.set_xscale('log')
    ax.set_xlim(1, 2000)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    legend_elements = [
        mpatches.Patch(facecolor='#4CAF50', label='Wheeled Robot'),
        mpatches.Patch(facecolor='#FF5722', label='Humanoid Robot'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    fig.savefig('fig2_control_frequency.png')
    plt.close(fig)
    print('Saved fig2_control_frequency.png')


def fig3_vla_timeline():
    """Timeline of VLA model releases."""
    models = [
        ('RT-1', 2022.9, 'wheeled', 35),
        ('SayCan', 2022.3, 'wheeled', None),
        ('PaLM-E', 2023.2, 'wheeled', 562000),
        ('RT-2', 2023.6, 'wheeled', 55000),
        ('Octo', 2024.1, 'cross', 93),
        ('OpenVLA', 2024.5, 'wheeled', 7000),
        ('HumanPlus', 2024.5, 'humanoid', None),
        ('RT-X', 2024.0, 'cross', 55000),
        ('HPT', 2024.7, 'cross', 1100),
        ('CrossFormer', 2024.8, 'cross', 130),
        ('pi0', 2024.8, 'cross', 3000),
        ('Helix', 2025.1, 'humanoid', 7000),
        ('GR00T N1', 2025.2, 'humanoid', 2200),
        ('WholeBodyVLA', 2025.8, 'humanoid', None),
        ('Gemini\nRobotics 2', 2026.7, 'cross', None),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    color_map = {
        'wheeled': '#4CAF50',
        'humanoid': '#FF5722',
        'cross': '#9C27B0',
    }
    label_map = {
        'wheeled': 'Wheeled / Arm',
        'humanoid': 'Humanoid',
        'cross': 'Cross-Embodiment',
    }

    for i, (name, date, cat, params) in enumerate(models):
        y_offset = 0.6 if i % 2 == 0 else -0.6
        ax.scatter(date, 0, s=120, c=color_map[cat], zorder=5, edgecolors='white', linewidth=1)
        ax.annotate(name, (date, 0), xytext=(0, 30 * (1 if i % 2 == 0 else -1)),
                    textcoords='offset points', ha='center', va='bottom' if i % 2 == 0 else 'top',
                    fontsize=8, fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

    ax.axhline(y=0, color='gray', linewidth=1.5, alpha=0.5)
    ax.set_xlim(2022.0, 2027.0)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('Year')
    ax.set_title('VLA Model Release Timeline (2022–2026)')
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    legend_elements = [mpatches.Patch(facecolor=c, label=l) for l, c in
                       [('Wheeled / Arm', '#4CAF50'), ('Humanoid', '#FF5722'), ('Cross-Embodiment', '#9C27B0')]]
    ax.legend(handles=legend_elements, loc='upper left')

    fig.savefig('fig3_vla_timeline.png')
    plt.close(fig)
    print('Saved fig3_vla_timeline.png')


def fig4_benchmark_landscape():
    """Scatter plot of benchmarks by domain and difficulty."""
    benchmarks = [
        ('LIBERO', 'manip', 130, 0.97, 'Franka (sim)'),
        ('CALVIN', 'manip', 34, 0.85, 'Franka (sim)'),
        ('SimplerEnv', 'manip', 30, 0.75, 'WidowX (sim)'),
        ('Meta-World', 'manip', 50, 0.57, 'Sawyer (sim)'),
        ('Habitat\nObjectNav', 'nav', 6, 0.65, 'Agent (sim)'),
        ('R2R-CE', 'nav', 7189, 0.54, 'Agent (sim)'),
        ('ALFRED', 'nav+manip', 7, 0.69, 'Agent (sim)'),
        ('OVMM', 'nav+manip', 1, 0.33, 'Stretch'),
        ('HumanoidBench', 'humanoid', 27, 0.15, 'H1 (sim)'),
    ]

    domain_colors = {
        'manip': '#2196F3',
        'nav': '#4CAF50',
        'nav+manip': '#FF9800',
        'humanoid': '#FF5722',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, domain, tasks, sota, robot in benchmarks:
        ax.scatter(tasks, sota * 100, s=200, c=domain_colors[domain],
                   edgecolors='white', linewidth=1.5, zorder=5, alpha=0.85)
        offset_x = 5 if tasks > 50 else 3
        offset_y = 3
        ax.annotate(name, (tasks, sota * 100),
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    fontsize=8, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Tasks (log scale)')
    ax.set_ylabel('Best Reported Success Rate (%)')
    ax.set_title('VLA Benchmark Landscape: Task Scale vs SOTA Performance')
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor='#2196F3', label='Manipulation'),
        mpatches.Patch(facecolor='#4CAF50', label='Navigation'),
        mpatches.Patch(facecolor='#FF9800', label='Nav + Manipulation'),
        mpatches.Patch(facecolor='#FF5722', label='Humanoid'),
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    fig.savefig('fig4_benchmark_landscape.png')
    plt.close(fig)
    print('Saved fig4_benchmark_landscape.png')


def fig5_data_availability():
    """Bar chart comparing data availability across domains."""
    domains = [
        'Autonomous\nDriving',
        'Arm\nManipulation\n(OXE)',
        'Mobile\nManipulation',
        'Quadruped\nLocomotion',
        'Humanoid\nManipulation',
        'Humanoid\nLoco-Manip',
    ]
    episodes_log = [9, 6, 4, 3.5, 2.7, 1.5]
    colors = ['#607D8B', '#2196F3', '#4CAF50', '#9C27B0', '#FF5722', '#FF5722']

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(domains, episodes_log, color=colors, edgecolor='white', linewidth=1.5)

    labels = ['Billions mi', '1M+ eps', '~10K eps', '~3K hrs', '~500 hrs', '<100 hrs']
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.15,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Data Scale (log₁₀ episodes/hours)')
    ax.set_title('Training Data Availability by Robot Domain')
    ax.set_ylim(0, 11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.axhline(y=4, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(5.5, 4.2, 'Minimum for reliable\ngeneralization', fontsize=8,
            color='red', alpha=0.7, ha='right')

    fig.savefig('fig5_data_availability.png')
    plt.close(fig)
    print('Saved fig5_data_availability.png')


def fig6_architecture_comparison():
    """Side-by-side architecture comparison diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    def draw_box(ax, x, y, w, h, text, color, fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2, text=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=2))
        if text:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, text, fontsize=7, color='#666', ha='left')

    # Wheeled VLA (left)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('Wheeled Robot VLA\n(Single End-to-End)', fontsize=13, fontweight='bold', pad=15)
    ax1.axis('off')

    draw_box(ax1, 1, 8.5, 3.5, 1, 'Camera\nImages', '#607D8B')
    draw_box(ax1, 5.5, 8.5, 3.5, 1, 'Language\nInstruction', '#607D8B')
    draw_arrow(ax1, 2.75, 8.5, 4.5, 7.3)
    draw_arrow(ax1, 7.25, 8.5, 5.5, 7.3)
    draw_box(ax1, 2, 5.8, 6, 1.5, 'VLM Backbone\n(7-55B params, 5-30 Hz)', '#1565C0')
    draw_arrow(ax1, 5, 5.8, 5, 5)
    draw_box(ax1, 2, 3.5, 6, 1.5, 'Action Head\n(Diffusion / AR Tokens)', '#0D47A1')
    draw_arrow(ax1, 5, 3.5, 5, 2.7)
    draw_box(ax1, 2, 1.2, 6, 1.5, 'PID Controller\n(100 Hz)', '#4CAF50')
    draw_arrow(ax1, 5, 1.2, 5, 0.5)
    draw_box(ax1, 2.5, -0.5, 5, 0.8, '7-DoF EEF + Base Vel', '#388E3C', fontsize=8)

    # Humanoid VLA (right)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('Humanoid VLA\n(Dual-System Hierarchical)', fontsize=13, fontweight='bold', pad=15)
    ax2.axis('off')

    draw_box(ax2, 1, 8.5, 3.5, 1, 'Camera\nImages', '#607D8B')
    draw_box(ax2, 5.5, 8.5, 3.5, 1, 'Language\nInstruction', '#607D8B')
    draw_arrow(ax2, 2.75, 8.5, 4.5, 7.8)
    draw_arrow(ax2, 7.25, 8.5, 5.5, 7.8)
    draw_box(ax2, 1.5, 6.3, 7, 1.5, 'System 2: VLM\n(7B params, 7-25 Hz)', '#E65100')
    draw_arrow(ax2, 5, 6.3, 5, 5.5, 'Latent\nvector')
    draw_box(ax2, 1.5, 4, 7, 1.5, 'System 1: Action Expert\n(80M params, 50-200 Hz)', '#BF360C')
    draw_arrow(ax2, 5, 4, 5, 3.2, 'Joint\ntargets')
    draw_box(ax2, 1.5, 1.7, 7, 1.5, 'Whole-Body Controller\n(RL-trained, 200-1000 Hz)', '#FF5722')
    draw_arrow(ax2, 5, 1.7, 5, 1)
    draw_box(ax2, 2, 0, 6, 0.8, '29-78 DoF Joint Torques', '#D32F2F', fontsize=8)

    fig.savefig('fig6_architecture_comparison.png')
    plt.close(fig)
    print('Saved fig6_architecture_comparison.png')


if __name__ == '__main__':
    fig1_action_space_comparison()
    fig2_control_frequency()
    fig3_vla_timeline()
    fig4_benchmark_landscape()
    fig5_data_availability()
    fig6_architecture_comparison()
    print('All figures generated.')
