import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import shutil
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def is_trivia_memory(memory):
    return "Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g [\"answer 1\", \"answer 2\", ...]" in memory.get('query', '')

def evaluate_memories(reference_data, input_data):
    results = []
    total_true_relevant = 0
    total_expected_relevant = 0
    total_irrelevant = 0
    total_trivia = 0
    total_memories = 0
    
    for ref_entry, input_entry in zip(reference_data, input_data):
        if ref_entry['query'] != input_entry['query']:
            continue
        
        true_relevant_queries = Counter(m['query'] for m in ref_entry['memories'])
        input_memories = Counter(m['query'] for m in input_entry['memories'])
        
        matched_relevant = sum(min(input_memories[query], count) for query, count in true_relevant_queries.items())
        expected_relevant = sum(true_relevant_queries.values())
        irrelevant = sum(count for query, count in input_memories.items() 
                         if query not in true_relevant_queries and not is_trivia_memory({'query': query}))
        trivia = sum(count for query, count in input_memories.items() if is_trivia_memory({'query': query}))
        
        total_true_relevant += matched_relevant
        total_expected_relevant += expected_relevant
        total_irrelevant += irrelevant
        total_trivia += trivia
        total_memories += sum(input_memories.values())
        
        results.append({
            'query': ref_entry['query'],
            'true_relevant': matched_relevant,
            'expected_relevant': expected_relevant,
            'true_relevant_percent': (matched_relevant / expected_relevant * 100) if expected_relevant > 0 else 0,
            'irrelevant': irrelevant,
            'trivia': trivia,
            'total_memories': sum(input_memories.values())
        })
    
    important_entries = len(results)
    recall_score = (total_true_relevant / total_expected_relevant) if total_expected_relevant > 0 else 0
    
    summary = {
        'recall_score': recall_score,
        'total_true_relevant_percent': (total_true_relevant / total_expected_relevant * 100) if total_expected_relevant > 0 else 0,
        'irrelevant_memories_percent': (total_irrelevant / total_memories * 100) if total_memories > 0 else 0,
        'total_trivia_memories_percent': (total_trivia / total_memories * 100) if total_memories > 0 else 0,
        'important_entries': important_entries
    }
    
    return results, summary


def plot_results(results, summary, output_path):
    # Prepare data
    df = pd.DataFrame(results)
    df['Entry'] = range(1, len(df) + 1)

    # Set up the plot with a white background
    plt.style.use('default')
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 2, height_ratios=[1, 0.5, 3], width_ratios=[1.2, 0.8])
    
    # Color palette for white background
    colors = {'True_relevant': '#4CAF50', 'Irrelevant': '#FFC107', 'Trivia': '#9C27B0', 'Missing_Relevant': '#F44336'}

    # Summary Statistics (top left)
    ax_summary = fig.add_subplot(gs[0, 0])
    summary_data = [
        ('Average\nRetrieved\nMemory', 100 - summary['irrelevant_memories_percent'] - summary['total_trivia_memories_percent'],
         summary['irrelevant_memories_percent'], summary['total_trivia_memories_percent']),
        ('Relevant\nInformation\nRetrieved', summary['total_true_relevant_percent'], 100 - summary['total_true_relevant_percent'])
    ]

    for i, (category, *values) in enumerate(summary_data):
        start = 0
        for j, value in enumerate(values):
            if category == 'Relevant\nInformation\nRetrieved':
                color = ['#4CAF50', '#F44336'][j]  # Green and Red for Relevant Memories
            else:
                color = list(colors.values())[j]
            ax_summary.barh(i, value, left=start, color=color, height=0.6, edgecolor='none')
            if value > 5:
                ax_summary.text(start + value/2, i, f'{value:.1f}%', ha='center', va='center', fontsize=12, color='black')
            start += value

    ax_summary.set_xlim(0, 100)
    ax_summary.set_xlabel('Percentage', fontsize=14)
    ax_summary.set_title('Dev Bench 2-2 Summary Statistics', fontsize=18, loc="right")
    ax_summary.set_yticks(range(len(summary_data)))
    ax_summary.set_yticklabels([x[0] for x in summary_data], fontsize=12)

    # Summary Statistics Printout (top right)
    ax_printout = fig.add_subplot(gs[0, 1])
    ax_printout.axis('off')
    printout_text = (
        f"Summary Statistics:\n"
        f"Total Relevant Retrieved %: {summary['total_true_relevant_percent']:.1f}%\n"
        f"Total Missing Relevant %: {100-(summary['total_true_relevant_percent']):.1f}%\n"
        f"Average Irrelevant %: {summary['irrelevant_memories_percent']:.1f}%\n"
        f"Average Trivia %: {summary['total_trivia_memories_percent']:.1f}%\n"
        f"No. Important Entries: {summary['important_entries']}\n\n"
        f"Key:\n"
        f"Total Relevant Retrieved %: Percentage of needed relevant memories retrieved\n"
        f"Total Missing Relevant %: Percentage of needed relevant memories NOT retrieved\n"
        f"Average Irrelevant %: Percentage of retrieved memories that were irrelevant for entry\n"
        f"Average Trivia %: Percentage of retrieved memories identified as trivia (filler)\n"
        f"No. Important Entries: The number of entries that comprise the testable content"
    )
    ax_printout.text(0.05, 0.95, printout_text, verticalalignment='top', fontsize=12, fontfamily='monospace', color='black')

    # # Legend (middle)
    # ax_legend = fig.add_subplot(gs[1, :])
    # ax_legend.axis('off')
    # legend_elements = [Patch(facecolor=color, edgecolor='none', label=label.replace('_', ' '))
    #                    for label, color in colors.items()]
    # legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', label='True Relevant Trend'))
    # ax_legend.legend(handles=legend_elements, loc='center', ncol=5, fontsize=12)

    # Memory Breakdown by Entry (bottom)
    ax_breakdown = fig.add_subplot(gs[1:3, :])

    bottoms = np.zeros(len(df))
    for category in ['true_relevant', 'irrelevant', 'trivia']:
        ax_breakdown.bar(df['Entry'], df[category], bottom=bottoms, color=colors[category.capitalize()], width=0.8, edgecolor='none')
        bottoms += df[category]

    # Add 'expected_relevant' as red caps on top of the bars
    ax_breakdown.bar(df['Entry'], df['expected_relevant'] - df['true_relevant'], 
                    bottom=df['true_relevant'], color=colors['Missing_Relevant'], width=0.8, edgecolor='none')

    ax_breakdown.set_xlabel('Entry Number', fontsize=14)
    ax_breakdown.set_ylabel('Number of Memories', fontsize=14)
    ax_breakdown.set_title('Memory Breakdown by Entry', fontsize=18)

    # Add total count labels on top of bars
    for i, total in enumerate(df['total_memories']):
        ax_breakdown.text(i+1, total, str(total), ha='center', va='bottom', fontsize=8, color='black')

    # Set x-axis ticks
    ax_breakdown.set_xticks(range(0, len(df)+1, 10))

    # Add gridlines
    ax_breakdown.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a smoothed trend line for true relevant memories
    x = df['Entry']
    y = df['true_relevant']
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    ax_breakdown.plot(x, p(x), linestyle='--', linewidth=1, color='black')

    # # Legend (middle)
    # ax_legend = fig.add_subplot(gs[1, :])
    # ax_legend.axis('off')
    # legend_elements = [Patch(facecolor=color, edgecolor='none', label=label.replace('_', ' '))
    #                    for label, color in colors.items()]
    # legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', label='True Relevant Trend'))
    # ax_legend.legend(handles=legend_elements, loc='center', ncol=5, fontsize=12)

    # Move legend inside the plot
    legend_elements = [Patch(facecolor=color, edgecolor='none', label=label.replace('_', ' '))
                    for label, color in colors.items()]
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', label='True Relevant Trend'))
    ax_breakdown.legend(handles=legend_elements, loc='upper left', ncol=5, fontsize=10, frameon=True, framealpha=0.8)

    # Add vertical padding
    y_max = max(df['total_memories'])
    ax_breakdown.set_ylim(0, y_max * 1.1)  # Add 10% padding above the highest bar

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Improved plot saved as '{output_path}'")
    plt.close()

def create_output_directory(base_path, recall_score):
    timestamp = int(time.time())
    dir_name = f"2-2_{recall_score:.2f}_{timestamp}"
    full_path = os.path.join(base_path, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path, timestamp

def write_results_to_file(results, summary, output_path):
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
        f.write("irrelevant_memories_percent: The percentage of retrieved memories that were irrelevant to the query.\n")
        f.write("total_trivia_memories_percent: The percentage of retrieved memories that were identified as trivia.\n")
        f.write("important_entries: The number of unique queries evaluated in this assessment.\n")
        
        f.write("\nSummary:\n")
        for key, value in summary.items():
            f.write(f"{key}: {value:.2f}\n")

def main():
    reference_data = load_json_file('./MLP_LTM/retrieval_assessment/reference_data/comparison_data_reference_2-2.json')
    input_data_path = './MLP_LTM/comparison_data/comparison_data.json'
    input_data = load_json_file(input_data_path)
    
    results, summary = evaluate_memories(reference_data, input_data)
    
    # Create output directory
    base_output_path = r"C:\Users\fkgde\Desktop\GoodAI\__FULLTIME\LTM-Benchmark\goodai-ltm-benchmark\MLP_LTM\_retrieval_data_output"
    output_dir, timestamp = create_output_directory(base_output_path, summary['recall_score'])
    
    # Rename and move input file
    new_input_filename = f"comparison_data_2-2_{summary['recall_score']:.2f}_{timestamp}.json"
    new_input_path = os.path.join(output_dir, new_input_filename)
    shutil.copy(input_data_path, new_input_path)
    print(f"Input data copied to: {new_input_path}")
    
    # Generate and save plot
    plot_filename = f"comparison_data_2-2_{summary['recall_score']:.2f}_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plot_results(results, summary, plot_path)
    
    # Write results to text file
    results_filename = f"results_2-2_{summary['recall_score']:.2f}_{timestamp}.txt"
    results_path = os.path.join(output_dir, results_filename)
    write_results_to_file(results, summary, results_path)
    print(f"Results written to: {results_path}")
    
    # Print results to terminal
    print("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        print(f"\nEntry {i}:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print("\nMetric Descriptions:")
    print("recall_score: The proportion of expected relevant memories that were successfully retrieved.")
    print("total_true_relevant_percent: The percentage of expected relevant memories that were retrieved, relative to the total expected.")
    print("irrelevant_memories_percent: The percentage of retrieved memories that were irrelevant to the query.")
    print("total_trivia_memories_percent: The percentage of retrieved memories that were identified as trivia.")
    print("important_entries: The number of unique queries evaluated in this assessment.")
    
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value:.2f}")
    
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()