import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import shutil
import seaborn as sns
import pandas as pd
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
            'total_memories': sum(input_memories.values()),
            'test': ref_entry['test'],
            'is_scored_question': ref_entry['is_scored_question']
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

def plot_original_results(results, summary, output_path):
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
    ax_summary.set_title('Dev Bench 4-1 Summary Statistics', fontsize=18, loc="right")
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

def plot_test_breakdown(results, output_path):
    df = pd.DataFrame(results)
    
    # Set up the plot
    plt.style.use('default')
    fig, axes = plt.subplots(5, 2, figsize=(24, 30))
    fig.suptitle('Test-by-Test Memory Breakdown', fontsize=20)
    
    # Color palette
    colors = {'True_relevant': '#4CAF50', 'Irrelevant': '#FFC107', 'Trivia': '#9C27B0', 'Missing_Relevant': '#F44336'}
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    test_types = sorted(df['test'].unique())
    test_types = [t for t in test_types if t != 'admin']
    
    for i, test_type in enumerate(test_types):
        ax = axes[i]
        test_df = df[df['test'] == test_type].copy()
        
        # Create a new 'Entry' column for this test
        test_df['Entry'] = range(1, len(test_df) + 1)
        
        bottoms = np.zeros(len(test_df))
        for category in ['true_relevant', 'irrelevant', 'trivia']:
            bars = ax.bar(test_df['Entry'], test_df[category], bottom=bottoms, 
                          color=colors[category.capitalize()], width=0.8, edgecolor='none')
            
            # Add outline for scored questions
            for bar, is_scored in zip(bars, test_df['is_scored_question']):
                if is_scored == 'yes':
                    bar.set_edgecolor('black')
                    bar.set_linewidth(2)
            
            bottoms += test_df[category]
        
        # Add 'expected_relevant' as red caps on top of the bars
        ax.bar(test_df['Entry'], test_df['expected_relevant'] - test_df['true_relevant'], 
               bottom=test_df['true_relevant'], color=colors['Missing_Relevant'], width=0.8, edgecolor='none')
        
        ax.set_title(f"{test_type} (n={len(test_df)})", fontsize=12)
        ax.set_xlabel('Entry Number', fontsize=10)
        ax.set_ylabel('Number of Memories', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Set x-axis ticks to show all entry numbers
        ax.set_xticks(test_df['Entry'])
        ax.set_xticklabels(test_df['Entry'], rotation=45, ha='right')
        
        # Add total count labels on top of bars
        for j, total in enumerate(test_df['total_memories']):
            ax.text(test_df['Entry'].iloc[j], total, str(total), ha='center', va='bottom', fontsize=8)
        
        # Set y-axis limit to be slightly higher than the maximum total memories
        ax.set_ylim(0, max(test_df['total_memories']) * 1.1)
    
    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add a legend
    legend_elements = [Patch(facecolor=color, edgecolor='none', label=label.replace('_', ' '))
                       for label, color in colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Adjust to make room for the main title
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Test breakdown plot saved as '{output_path}'")
    plt.close()


def create_output_directory(base_path, recall_score):
    timestamp = int(time.time())
    dir_name = f"4-1_{recall_score:.2f}_{timestamp}"
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

def create_output_directory(base_path, recall_score):
    timestamp = int(time.time())
    dir_name = f"4-1_{recall_score:.2f}_{timestamp}"
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

def calculate_enhanced_summary(results):
    df = pd.DataFrame(results)
    
    test_summaries = df.groupby('test').agg({
        'true_relevant': 'sum',
        'expected_relevant': 'sum',
        'irrelevant': 'sum',
        'trivia': 'sum',
        'total_memories': 'sum',
        'is_scored_question': lambda x: (x == 'yes').sum()
    }).reset_index()
    
    test_summaries['recall_score'] = test_summaries['true_relevant'] / test_summaries['expected_relevant']
    test_summaries['irrelevant_percent'] = test_summaries['irrelevant'] / test_summaries['total_memories'] * 100
    test_summaries['trivia_percent'] = test_summaries['trivia'] / test_summaries['total_memories'] * 100
    
    overall_recall = df['true_relevant'].sum() / df['expected_relevant'].sum()
    overall_irrelevant = df['irrelevant'].sum() / df['total_memories'].sum() * 100
    overall_trivia = df['trivia'].sum() / df['total_memories'].sum() * 100
    
    scored_questions = df[df['is_scored_question'] == 'yes']
    scored_recall = scored_questions['true_relevant'].sum() / scored_questions['expected_relevant'].sum()
    
    enhanced_summary = {
        'overall_recall_score': overall_recall,
        'overall_irrelevant_percent': overall_irrelevant,
        'overall_trivia_percent': overall_trivia,
        'scored_questions_recall': scored_recall,
        'test_summaries': test_summaries.to_dict('records'),
        'total_entries': len(df),
        'total_scored_entries': len(scored_questions),
        'total_memories': df['total_memories'].sum(),
        'average_memories_per_entry': df['total_memories'].mean()
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

def main():
    reference_data = load_json_file('./MLP_LTM/retrieval_assessment/reference_data/comparison_data_reference_enhanced_4-1.json')
    input_data_path = './MLP_LTM/comparison_data/comparison_data_prompt-02_7.3.json'
    input_data = load_json_file(input_data_path)
    
    results, summary = evaluate_memories(reference_data, input_data)
    
    # Create output directory
    base_output_path = r"C:\Users\fkgde\Desktop\GoodAI\__FULLTIME\LTM-Benchmark\goodai-ltm-benchmark\MLP_LTM\_retrieval_data_output"
    output_dir, timestamp = create_output_directory(base_output_path, summary['recall_score'])
    
    # Rename and move input file
    new_input_filename = f"comparison_data_4-1_{summary['recall_score']:.2f}_{timestamp}.json"
    new_input_path = os.path.join(output_dir, new_input_filename)
    shutil.copy(input_data_path, new_input_path)
    print(f"Input data copied to: {new_input_path}")
    
    # Generate and save original plot
    original_plot_filename = f"comparison_data_4-1_{summary['recall_score']:.2f}_{timestamp}_original.png"
    original_plot_path = os.path.join(output_dir, original_plot_filename)
    plot_original_results(results, summary, original_plot_path)
    
    # Generate and save test breakdown plot
    test_plot_filename = f"comparison_data_4-1_{summary['recall_score']:.2f}_{timestamp}_test_breakdown.png"
    test_plot_path = os.path.join(output_dir, test_plot_filename)
    plot_test_breakdown(results, test_plot_path)
    
    # Write results to text file
    results_filename = f"results_4-1_{summary['recall_score']:.2f}_{timestamp}.txt"
    results_path = os.path.join(output_dir, results_filename)
    write_results_to_file(results, summary, results_path)
    print(f"Results written to: {results_path}")
    
    # Calculate and print enhanced summary
    enhanced_summary = calculate_enhanced_summary(results)
    print_enhanced_summary(enhanced_summary)
    
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()