import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats
import time

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def is_trivia_memory(memory):
    return "Here are some trivia questions and answers for you to process." in memory.get('query', '')

def evaluate_memories(reference_data, input_data):
    results = []
    total_true_relevant = 0
    total_expected_relevant = 0
    total_irrelevant = 0
    total_trivia = 0
    total_retrieved = 0
    total_filtered = 0

    for ref_entry, input_entry in zip(reference_data, input_data):
        if ref_entry['query'] != input_entry['query']:
            continue

        true_relevant_queries = set(m['query'] for m in ref_entry['memories'])
        input_memories = [m['query'] for m in input_entry['memories']]

        matched_relevant = sum(1 for query in input_memories if query in true_relevant_queries)
        expected_relevant = len(true_relevant_queries)
        irrelevant = sum(1 for query in input_memories if query not in true_relevant_queries and not is_trivia_memory({'query': query}))
        trivia = sum(1 for query in input_memories if is_trivia_memory({'query': query}))

        retrieved_count = input_entry.get('retrieved_memories_count', len(input_memories))
        filtered_count = len(input_memories)

        total_true_relevant += matched_relevant
        total_expected_relevant += expected_relevant
        total_irrelevant += irrelevant
        total_trivia += trivia
        total_retrieved += retrieved_count
        total_filtered += filtered_count

        results.append({
            'query': ref_entry['query'],
            'true_relevant': matched_relevant,
            'expected_relevant': expected_relevant,
            'true_relevant_percent': (matched_relevant / expected_relevant * 100) if expected_relevant > 0 else 0,
            'irrelevant': irrelevant,
            'trivia': trivia,
            'total_memories': filtered_count,
            'test': ref_entry['test'],
            'is_scored_question': ref_entry['is_scored_question'],
            'retrieved_memories_count': retrieved_count,
            'filtered_memories_count': filtered_count
        })

    important_entries = len(results)
    recall_score = (total_true_relevant / total_expected_relevant) if total_expected_relevant > 0 else 0

    summary = {
        'recall_score': recall_score,
        'total_true_relevant_percent': (total_true_relevant / total_expected_relevant * 100) if total_expected_relevant > 0 else 0,
        'irrelevant_memories_percent': (total_irrelevant / total_filtered * 100) if total_filtered > 0 else 0,
        'total_trivia_memories_percent': (total_trivia / total_filtered * 100) if total_filtered > 0 else 0,
        'important_entries': important_entries,
        'total_retrieved': total_retrieved,
        'total_filtered': total_filtered
    }

    return results, summary

def plot_original_results(results, summary, output_path):
    df = pd.DataFrame(results)
    df['Entry'] = range(1, len(df) + 1)

    plt.style.use('default')
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 2, height_ratios=[1, 0.5, 3], width_ratios=[1.2, 0.8])

    colors = {'True_relevant': '#4CAF50', 'Irrelevant': '#FFC107', 'Trivia': '#9C27B0', 'Missing_Relevant': '#F44336'}

    ax_summary = fig.add_subplot(gs[0, 0])
    summary_data = [
        ('Average\nFiltered\nMemory', 100 - summary['irrelevant_memories_percent'] - summary['total_trivia_memories_percent'],
         summary['irrelevant_memories_percent'], summary['total_trivia_memories_percent']),
        ('Relevant\nInformation\nRetrieved', summary['total_true_relevant_percent'], 100 - summary['total_true_relevant_percent'])
    ]

    for i, (category, *values) in enumerate(summary_data):
        start = 0
        for j, value in enumerate(values):
            if category == 'Relevant\nInformation\nRetrieved':
                color = ['#4CAF50', '#F44336'][j]
            else:
                color = list(colors.values())[j]
            ax_summary.barh(i, value, left=start, color=color, height=0.6, edgecolor='none')
            if value > 5:
                ax_summary.text(start + value/2, i, f'{value:.1f}%', ha='center', va='center', fontsize=12, color='black')
            start += value

    ax_summary.set_xlim(0, 100)
    ax_summary.set_xlabel('Percentage', fontsize=14)
    ax_summary.set_title('Dev Bench 4-1 Summary Statistics', fontsize=18, loc="right")
    ax_summary.set_yticks(range(len(summary_data)))
    ax_summary.set_yticklabels([x[0] for x in summary_data], fontsize=12)

    ax_printout = fig.add_subplot(gs[0, 1])
    ax_printout.axis('off')
    printout_text = (
        f"Summary Statistics:\n"
        f"Total Relevant Retrieved %: {summary['total_true_relevant_percent']:.1f}%\n"
        f"Total Missing Relevant %: {100-(summary['total_true_relevant_percent']):.1f}%\n"
        f"Average Irrelevant %: {summary['irrelevant_memories_percent']:.1f}%\n"
        f"Average Trivia %: {summary['total_trivia_memories_percent']:.1f}%\n"
        f"No. Important Entries: {summary['important_entries']}\n"
        f"Total Retrieved Memories: {summary['total_retrieved']}\n"
        f"Total Filtered Memories: {summary['total_filtered']}\n\n"
        f"Key:\n"
        f"Total Relevant Retrieved %: Percentage of needed relevant memories retrieved\n"
        f"Total Missing Relevant %: Percentage of needed relevant memories NOT retrieved\n"
        f"Average Irrelevant %: Percentage of filtered memories that were irrelevant for entry\n"
        f"Average Trivia %: Percentage of filtered memories identified as trivia (filler)\n"
        f"No. Important Entries: The number of entries that comprise the testable content"
    )
    ax_printout.text(0.05, 0.95, printout_text, verticalalignment='top', fontsize=12, fontfamily='monospace', color='black')

    ax_breakdown = fig.add_subplot(gs[1:3, :])

    bottoms = np.zeros(len(df))
    for category in ['true_relevant', 'irrelevant', 'trivia']:
        ax_breakdown.bar(df['Entry'], df[category], bottom=bottoms, color=colors[category.capitalize()], width=0.8, edgecolor='none')
        bottoms += df[category]

    ax_breakdown.bar(df['Entry'], df['expected_relevant'] - df['true_relevant'],
                    bottom=df['true_relevant'], color=colors['Missing_Relevant'], width=0.8, edgecolor='none')

    ax_breakdown.set_xlabel('Entry Number', fontsize=14)
    ax_breakdown.set_ylabel('Number of Memories', fontsize=14)
    ax_breakdown.set_title('Memory Breakdown by Entry', fontsize=18)

    for i, total in enumerate(df['total_memories']):
        ax_breakdown.text(i+1, total, str(total), ha='center', va='bottom', fontsize=8, color='black')

    ax_breakdown.set_xticks(range(0, len(df)+1, 10))
    ax_breakdown.grid(axis='y', linestyle='--', alpha=0.7)

    x = df['Entry']
    y = df['true_relevant']
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    ax_breakdown.plot(x, p(x), linestyle='--', linewidth=1, color='black')

    legend_elements = [Patch(facecolor=color, edgecolor='none', label=label.replace('_', ' '))
                    for label, color in colors.items()]
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', label='True Relevant Trend'))
    ax_breakdown.legend(handles=legend_elements, loc='upper left', ncol=5, fontsize=10, frameon=True, framealpha=0.8)

    y_max = max(df['total_memories'])
    ax_breakdown.set_ylim(0, y_max * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Improved plot saved as '{output_path}'")
    plt.close()

def plot_test_breakdown(results, output_path):
    df = pd.DataFrame(results)

    plt.style.use('default')
    fig, axes = plt.subplots(5, 2, figsize=(24, 30))
    fig.suptitle('Test-by-Test Memory Breakdown', fontsize=20)

    colors = {'True_relevant': '#4CAF50', 'Irrelevant': '#FFC107', 'Trivia': '#9C27B0', 'Missing_Relevant': '#F44336'}

    axes = axes.flatten()

    test_types = sorted(df['test'].unique())
    test_types = [t for t in test_types if t != 'admin']

    for i, test_type in enumerate(test_types):
        ax = axes[i]
        test_df = df[df['test'] == test_type].copy()

        test_df['Entry'] = range(1, len(test_df) + 1)

        bottoms = np.zeros(len(test_df))
        for category in ['true_relevant', 'irrelevant', 'trivia']:
            bars = ax.bar(test_df['Entry'], test_df[category], bottom=bottoms,
                          color=colors[category.capitalize()], width=0.8, edgecolor='none')

            for bar, is_scored in zip(bars, test_df['is_scored_question']):
                if is_scored == 'yes':
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)

            bottoms += test_df[category]

        ax.bar(test_df['Entry'], test_df['expected_relevant'] - test_df['true_relevant'],
               bottom=test_df['true_relevant'], color=colors['Missing_Relevant'], width=0.8, edgecolor='none')

        ax.set_title(f"{test_type} (n={len(test_df)})", fontsize=12)
        ax.set_xlabel('Entry Number', fontsize=10)
        ax.set_ylabel('Number of Memories', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)

        ax.set_xticks(test_df['Entry'])
        ax.set_xticklabels(test_df['Entry'], rotation=45, ha='right')

        for j, total in enumerate(test_df['total_memories']):
            ax.text(test_df['Entry'].iloc[j], total, str(total), ha='center', va='bottom', fontsize=8)

        ax.set_ylim(0, max(test_df['total_memories']) * 1.1)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    legend_elements = [Patch(facecolor=color, edgecolor='none', label=label.replace('_', ' '))
                       for label, color in colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Test breakdown plot saved as '{output_path}'")
    plt.close()

def plot_retrieved_vs_filtered(results, output_path):
    df = pd.DataFrame(results)
    df['Entry'] = range(1, len(df) + 1)

    plt.figure(figsize=(20, 10))
    plt.bar(df['Entry'], df['retrieved_memories_count'], label='Retrieved', alpha=0.7)
    plt.bar(df['Entry'], df['filtered_memories_count'], label='Filtered', alpha=0.7)

    plt.xlabel('Entry Number', fontsize=14)
    plt.ylabel('Number of Memories', fontsize=14)
    plt.title('Retrieved vs Filtered Memories by Entry', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks(range(0, len(df)+1, 10))

    for i, (retrieved, filtered) in enumerate(zip(df['retrieved_memories_count'], df['filtered_memories_count'])):
        plt.text(i+1, retrieved, str(retrieved), ha='center', va='bottom', fontsize=8)
        plt.text(i+1, filtered, str(filtered), ha='center', va='bottom', fontsize=8)

    plt.ylim(0, max(df['retrieved_memories_count']) * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Retrieved vs Filtered plot saved as '{output_path}'")
    plt.close()

def calculate_enhanced_summary(results):
    df = pd.DataFrame(results)

    test_summaries = df.groupby('test').agg({
        'true_relevant': 'sum',
        'expected_relevant': 'sum',
        'irrelevant': 'sum',
        'trivia': 'sum',
        'total_memories': 'sum',
        'is_scored_question': lambda x: (x == 'yes').sum(),
        'retrieved_memories_count': 'sum',
        'filtered_memories_count': 'sum'
    }).reset_index()

    test_summaries['recall_score'] = test_summaries['true_relevant'] / test_summaries['expected_relevant']
    test_summaries['irrelevant_percent'] = test_summaries['irrelevant'] / test_summaries['filtered_memories_count'] * 100
    test_summaries['trivia_percent'] = test_summaries['trivia'] / test_summaries['filtered_memories_count'] * 100

    overall_recall = df['true_relevant'].sum() / df['expected_relevant'].sum()
    overall_irrelevant = df['irrelevant'].sum() / df['filtered_memories_count'].sum() * 100
    overall_trivia = df['trivia'].sum() / df['filtered_memories_count'].sum() * 100

    scored_questions = df[df['is_scored_question'] == 'yes']
    scored_recall = scored_questions['true_relevant'].sum() / scored_questions['expected_relevant'].sum()

    df['retrieval_efficiency'] = df['filtered_memories_count'] / df['retrieved_memories_count']

    retrieved_filtered_summary = {
        'mean_retrieved': df['retrieved_memories_count'].mean(),
        'sd_retrieved': df['retrieved_memories_count'].std(),
        'mean_filtered': df['filtered_memories_count'].mean(),
        'sd_filtered': df['filtered_memories_count'].std(),
        'mean_efficiency': df['retrieval_efficiency'].mean(),
        'sd_efficiency': df['retrieval_efficiency'].std(),
        'total_retrieved': df['retrieved_memories_count'].sum(),
        'total_filtered': df['filtered_memories_count'].sum(),
        'overall_efficiency': df['filtered_memories_count'].sum() / df['retrieved_memories_count'].sum()
    }

    enhanced_summary = {
        'overall_recall_score': overall_recall,
        'overall_irrelevant_percent': overall_irrelevant,
        'overall_trivia_percent': overall_trivia,
        'scored_questions_recall': scored_recall,
        'test_summaries': test_summaries.to_dict('records'),
        'total_entries': len(df),
        'total_scored_entries': len(scored_questions),
        'total_memories': df['total_memories'].sum(),
        'average_memories_per_entry': df['total_memories'].mean(),
        'retrieved_filtered_summary': retrieved_filtered_summary
    }

    return enhanced_summary

def print_enhanced_summary(enhanced_summary):
    print("\nEnhanced Summary:")
    print(f"Overall Recall Score: {enhanced_summary['overall_recall_score']:.2f}")
    print(f"Overall Irrelevant %: {enhanced_summary['overall_irrelevant_percent']:.2f}%")
    print(f"Overall Trivia %: {enhanced_summary['overall_trivia_percent']:.2f}%")
    print(f"Scored Questions Recall: {enhanced_summary['scored_questions_recall']:.2f}")
    print(f"Total Entries: {enhanced_summary['total_entries']}")
    print(f"Total Scored Entries: {enhanced_summary['total_scored_entries']}")
    print(f"Total Memories: {enhanced_summary['total_memories']}")
    print(f"Average Memories per Entry: {enhanced_summary['average_memories_per_entry']:.2f}")

    print("\nTest-specific Summaries:")
    for test_summary in enhanced_summary['test_summaries']:
        print(f"\n{test_summary['test']}:")
        print(f"  Recall Score: {test_summary['recall_score']:.2f}")
        print(f"  Irrelevant %: {test_summary['irrelevant_percent']:.2f}%")
        print(f"  Trivia %: {test_summary['trivia_percent']:.2f}%")
        print(f"  Scored Questions: {test_summary['is_scored_question']}")

    print("\nRetrieved vs Filtered Memories Summary:")
    rf_summary = enhanced_summary['retrieved_filtered_summary']
    print(f"Mean Retrieved Memories: {rf_summary['mean_retrieved']:.2f} (SD: {rf_summary['sd_retrieved']:.2f})")
    print(f"Mean Filtered Memories: {rf_summary['mean_filtered']:.2f} (SD: {rf_summary['sd_filtered']:.2f})")
    print(f"Mean Retrieval Efficiency: {rf_summary['mean_efficiency']:.2f} (SD: {rf_summary['sd_efficiency']:.2f})")
    print(f"Total Retrieved Memories: {rf_summary['total_retrieved']}")
    print(f"Total Filtered Memories: {rf_summary['total_filtered']}")
    print(f"Overall Retrieval Efficiency: {rf_summary['overall_efficiency']:.2f}")

def write_results_to_file(results, summary, enhanced_summary, output_path):
    with open(output_path, 'w') as f:
        f.write("Detailed Results:\n")
        for i, result in enumerate(results, 1):
            f.write(f"\nEntry {i}:\n")
            for key, value in result.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.2f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

        f.write("\nMetric Descriptions:\n")
        f.write("recall_score: The proportion of expected relevant memories that were successfully retrieved.\n")
        f.write("total_true_relevant_percent: The percentage of expected relevant memories that were retrieved, relative to the total expected.\n")
        f.write("irrelevant_memories_percent: The percentage of filtered memories that were irrelevant to the query.\n")
        f.write("total_trivia_memories_percent: The percentage of filtered memories that were identified as trivia.\n")
        f.write("important_entries: The number of unique queries evaluated in this assessment.\n")

        f.write("\nSummary:\n")
        for key, value in summary.items():
            f.write(f"{key}: {value:.2f}\n")

        f.write("\nEnhanced Summary:\n")
        f.write(f"Overall Recall Score: {enhanced_summary['overall_recall_score']:.2f}\n")
        f.write(f"Overall Irrelevant %: {enhanced_summary['overall_irrelevant_percent']:.2f}%\n")
        f.write(f"Overall Trivia %: {enhanced_summary['overall_trivia_percent']:.2f}%\n")
        f.write(f"Scored Questions Recall: {enhanced_summary['scored_questions_recall']:.2f}\n")
        f.write(f"Total Entries: {enhanced_summary['total_entries']}\n")
        f.write(f"Total Scored Entries: {enhanced_summary['total_scored_entries']}\n")
        f.write(f"Total Memories: {enhanced_summary['total_memories']}\n")
        f.write(f"Average Memories per Entry: {enhanced_summary['average_memories_per_entry']:.2f}\n")

        f.write("\nTest-specific Summaries:\n")
        for test_summary in enhanced_summary['test_summaries']:
            f.write(f"\n{test_summary['test']}:\n")
            f.write(f"  Recall Score: {test_summary['recall_score']:.2f}\n")
            f.write(f"  Irrelevant %: {test_summary['irrelevant_percent']:.2f}%\n")
            f.write(f"  Trivia %: {test_summary['trivia_percent']:.2f}%\n")
            f.write(f"  Scored Questions: {test_summary['is_scored_question']}\n")

        f.write("\nRetrieved vs Filtered Memories Summary:\n")
        rf_summary = enhanced_summary['retrieved_filtered_summary']
        f.write(f"Mean Retrieved Memories: {rf_summary['mean_retrieved']:.2f} (SD: {rf_summary['sd_retrieved']:.2f})\n")
        f.write(f"Mean Filtered Memories: {rf_summary['mean_filtered']:.2f} (SD: {rf_summary['sd_filtered']:.2f})\n")
        f.write(f"Mean Retrieval Efficiency: {rf_summary['mean_efficiency']:.2f} (SD: {rf_summary['sd_efficiency']:.2f})\n")
        f.write(f"Total Retrieved Memories: {rf_summary['total_retrieved']}\n")
        f.write(f"Total Filtered Memories: {rf_summary['total_filtered']}\n")
        f.write(f"Overall Retrieval Efficiency: {rf_summary['overall_efficiency']:.2f}\n")

        retrieved = [r['retrieved_memories_count'] for r in results]
        filtered = [r['filtered_memories_count'] for r in results]
        t_stat, p_value = stats.ttest_rel(retrieved, filtered)
        f.write(f"\nPaired t-test results (Retrieved vs Filtered):\n")
        f.write(f"t-statistic: {t_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")

def get_project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent

def load_json_file(file_path):
    """Load and return the contents of a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def create_output_directory(base_path, recall_score):
    """Create and return the path to a new output directory."""
    timestamp = int(time.time())
    dir_name = f"4-1_{recall_score:.2f}_{timestamp}"
    full_path = os.path.join(base_path, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path, timestamp

def main():
    # Get the project root directory
    project_root = get_project_root()

    # Construct file paths using os.path.join for OS-agnostic paths
    reference_data_path = os.path.join(project_root, "retrieval_assessment", "reference_data", "comparison_data_reference_enhanced_4-1.json")
    input_data_path = os.path.join(project_root, "comparison_data", "comparison_data_prompt-02_7.3.json")

    # Load data
    reference_data = load_json_file(reference_data_path)
    input_data = load_json_file(input_data_path)

    results, summary = evaluate_memories(reference_data, input_data)

    # Update the base output path
    base_output_path = os.path.join(project_root, "_retrieval_data_output")

    # Create output directory
    output_dir, timestamp = create_output_directory(base_output_path, summary['recall_score'])

    # Update file paths for output
    new_input_filename = f"comparison_data_4-1_{summary['recall_score']:.2f}_{timestamp}.json"
    new_input_path = os.path.join(output_dir, new_input_filename)
    os.makedirs(os.path.dirname(new_input_path), exist_ok=True)
    shutil.copy(input_data_path, new_input_path)

    new_input_filename = f"comparison_data_4-1_{summary['recall_score']:.2f}_{timestamp}.json"
    new_input_path = os.path.join(output_dir, new_input_filename)
    shutil.copy(input_data_path, new_input_path)
    print(f"Input data copied to: {new_input_path}")

    original_plot_filename = f"comparison_data_4-1_{summary['recall_score']:.2f}_{timestamp}_original.png"
    original_plot_path = os.path.join(output_dir, original_plot_filename)
    plot_original_results(results, summary, original_plot_path)

    test_plot_filename = f"comparison_data_4-1_{summary['recall_score']:.2f}_{timestamp}_test_breakdown.png"
    test_plot_path = os.path.join(output_dir, test_plot_filename)
    plot_test_breakdown(results, test_plot_path)

    rf_plot_filename = f"comparison_data_4-1_{summary['recall_score']:.2f}_{timestamp}_retrieved_filtered.png"
    rf_plot_path = os.path.join(output_dir, rf_plot_filename)
    plot_retrieved_vs_filtered(results, rf_plot_path)

    enhanced_summary = calculate_enhanced_summary(results)
    print_enhanced_summary(enhanced_summary)

    results_filename = f"results_4-1_{summary['recall_score']:.2f}_{timestamp}.txt"
    results_path = os.path.join(output_dir, results_filename)
    write_results_to_file(results, summary, enhanced_summary, results_path)
    print(f"Results written to: {results_path}")

    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
