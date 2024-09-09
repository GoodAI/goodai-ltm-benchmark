import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats
import time
import logging
from fuzzywuzzy import fuzz
from collections import defaultdict
from typing import List, Dict, Any, Union
import argparse
from datetime import datetime
import shutil
from goodai.ltm.mem.base import RetrievedMemory

from utils.constants import RETRIEVAL_REFERENCE_DIR, DATA_DIR
from utils.util_math import get_dimensions


class FileManager:
    def __init__(self, base_directory: Path):
        self.base_directory = Path(base_directory)
        self.ensure_directory_exists(self.base_directory)
        self.comparison_data_file = self.base_directory / "comparison_data.json"

    def ensure_directory_exists(self, path):
        path.mkdir(parents=True, exist_ok=True)

    def generate_filename(self, prefix="comparison_data"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.json"

    def clear_comparison_data(self, output_dir: Path, timestamp: int):
        if self.comparison_data_file.exists():
            new_filename = f"old_comparison_data_{datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')}.json"
            new_file_path = output_dir / new_filename

            shutil.move(str(self.comparison_data_file), str(new_file_path))

            print(f"Previous comparison data moved to {new_file_path}")

        with open(self.comparison_data_file, 'w') as f:
            json.dump([], f)

        print(f"Created new empty comparison data file at {self.comparison_data_file}")

    def write_data(self, data):
        try:
            with open(self.comparison_data_file, 'w', encoding="utf-8", errors="ignore") as f:
                json.dump(data, f, indent=2)
            print(f"Data written to {self.comparison_data_file}")
        except PermissionError as e:
            print(f"Permission error when writing to {self.comparison_data_file}. Error: {e}")
            fallback_path = Path.home() / "comparison_data_fallback.json"
            with open(fallback_path, 'w',  encoding="utf-8", errors="ignore") as f:
                json.dump(data, f, indent=2)
            print(f"Data written to fallback location: {fallback_path}")

    def read_data(self):
        if not self.comparison_data_file.exists():
            print(f"Comparison data file not found at {self.comparison_data_file}")
            return []
        with open(self.comparison_data_file, 'r',  encoding="utf-8", errors="ignore") as f:
            return json.load(f)

class RetrievalEvaluator:
    def __init__(self):
        self.project_root = DATA_DIR.joinpath("retrieval_evaluator")
        self.dev_bench_reference_data_path = RETRIEVAL_REFERENCE_DIR
        self.comparison_data_path = self.project_root / "comparison_data"
        self.logs_path = self.project_root / "logs"
        self.evaluation_outputs_path = self.project_root / "evaluation_outputs"
        self.logger = logging.getLogger("retrieval_evaluator")
        self.results: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self.enhanced_summary: Dict[str, Any] = {}

        self.file_manager = FileManager(self.comparison_data_path)

        for path in [self.project_root, self.dev_bench_reference_data_path, self.comparison_data_path,
                     self.logs_path, self.evaluation_outputs_path]:
            path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self, benchmark_version: str) -> None:
        log_file = self.logs_path / f'evaluation_{benchmark_version}.log'
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def clear_comparison_data(self) -> None:
        self.file_manager.clear_comparison_data()
        self.logger.info(f"Cleared previous comparison data and created new file at {self.file_manager.comparison_data_file}")


    def load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            self.logger.warning(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON from file: {file_path}")
            return []

    def is_trivia_memory(self, memory: Dict[str, Any]) -> bool:
        return "Here are some trivia questions and answers for you to process." in memory.get('query', '')

    def fuzzy_match(self, query1: str, query2: str, threshold: int = 85) -> bool:
        return fuzz.ratio(query1.lower(), query2.lower()) >= threshold

    def evaluate_memories(self, reference_data: List[Dict[str, Any]], input_data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        results: list[Dict[str, Any]] = []
        total_true_relevant: int = 0
        total_expected_relevant: int = 0
        total_irrelevant: int = 0
        total_trivia: int = 0
        total_retrieved: int = 0
        total_filtered: int = 0

        input_dict = defaultdict(list)
        for entry in input_data:
            input_dict[self._get_query_key(entry['query'])].append(entry)

        for i, ref_entry in enumerate(reference_data, 1):
            self.logger.info(f"\nProcessing reference entry {i}:")
            self.logger.info(f"Reference query: {ref_entry['query']}")
            self.logger.info(f"Test type: {ref_entry['test']}")
            self.logger.info(f"Is scored: {ref_entry['is_scored_question']}")

            matched_input = None
            for query, entries in input_dict.items():
                if self.fuzzy_match(self._get_query_key(ref_entry['query']), query):
                    matched_input = entries[0]
                    break

            if matched_input is None:
                self.logger.warning(f"No matching input entry found for query: {ref_entry['query']}. Using placeholder.")
                matched_input = {'query': ref_entry['query'], 'memories': []}

            true_relevant_queries = set(m['query'] for m in ref_entry['memories'])
            input_memories = [m['query'] for m in matched_input.get('memories', [])]

            matched_relevant = sum(1 for query in input_memories if any(self.fuzzy_match(query, true_query) for true_query in true_relevant_queries))
            expected_relevant = len(true_relevant_queries)
            irrelevant = sum(1 for query in input_memories if not any(self.fuzzy_match(query, true_query) for true_query in true_relevant_queries) and not self.is_trivia_memory({'query': query}))
            trivia = sum(1 for query in input_memories if self.is_trivia_memory({'query': query}))

            retrieved_count = matched_input.get('retrieved_memories_count', len(input_memories))
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
                'filtered_memories_count': filtered_count,
                'timestamp': ref_entry['timestamp']
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

    def plot_original_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any], output_path: Path, benchmark_version: str) -> None:

        # Sort by reference timestamp
        results = sorted(results, key=lambda x: x["timestamp"])

        df = pd.DataFrame(results)
        df['Entry'] = range(1, len(df) + 1)

        plt.style.use('default')
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(3, 2, height_ratios=[1, 0.5, 3], width_ratios=[1.2, 0.8])

        colors = {'true_relevant': '#4CAF50', 'irrelevant': '#FFC107', 'trivia': '#9C27B0', 'missing_relevant': '#F44336'}

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
        ax_summary.set_title(f'`{benchmark_version}` Summary Statistics', fontsize=18, loc="right")
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
            ax_breakdown.bar(df['Entry'], df[category], bottom=bottoms, color=colors[category], width=0.8, edgecolor='none')
            bottoms += df[category]

        ax_breakdown.bar(df['Entry'], df['expected_relevant'] - df['true_relevant'],
                        bottom=df['true_relevant'], color=colors['missing_relevant'], width=0.8, edgecolor='none')

        ax_breakdown.set_xlabel('Test message', fontsize=14)
        ax_breakdown.set_ylabel('Number of Memories', fontsize=14)
        ax_breakdown.set_title('Memory Breakdown by Entry', fontsize=18)

        for i, total in enumerate(df['total_memories']):
            ax_breakdown.text(i+1, total, str(total), ha='center', va='bottom', fontsize=8, color='black')

        x_tics_text = [x['test'] + (" *Q*" if x["is_scored_question"] else "") for x in results]
        ax_breakdown.set_xticks(range(1, len(x_tics_text) + 1))
        ax_breakdown.set_xticklabels(x_tics_text, minor=False, rotation=90)
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
        self.logger.info(f"Original plot saved as '{output_path}'")
        plt.close()

    def plot_test_breakdown(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        df = pd.DataFrame(results)

        plt.style.use('default')

        test_types = sorted(df['test'].unique())
        test_types = [t for t in test_types if t != 'admin']
        dims = get_dimensions(len(test_types))
        rows = max(dims)
        cols = min(dims)

        fig, axes = plt.subplots(rows, cols, figsize=(24, 30))
        fig.suptitle('Test-by-Test Memory Breakdown', fontsize=20)

        colors = {'true_relevant': '#4CAF50', 'irrelevant': '#FFC107', 'trivia': '#9C27B0', 'missing_relevant': '#F44336'}

        axes = axes.flatten()

        for i, test_type in enumerate(test_types):
            ax = axes[i]
            test_df = df[df['test'] == test_type].copy()

            test_df['Entry'] = range(1, len(test_df) + 1)

            bottoms = np.zeros(len(test_df))
            for category in ['true_relevant', 'irrelevant', 'trivia']:
                bars = ax.bar(test_df['Entry'], test_df[category], bottom=bottoms,
                            color=colors[category.lower()], width=0.8, edgecolor='none')

                for bar, is_scored in zip(bars, test_df['is_scored_question']):
                    if is_scored == 'yes':
                        bar.set_edgecolor('black')
                        bar.set_linewidth(2)

                bottoms += test_df[category]

            ax.bar(test_df['Entry'], test_df['expected_relevant'] - test_df['true_relevant'],
                bottom=test_df['true_relevant'], color=colors['missing_relevant'], width=0.8, edgecolor='none')

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

    def plot_retrieved_vs_filtered(self, results: List[Dict[str, Any]], output_path: Path) -> None:
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
        self.logger.info(f"Retrieved vs Filtered plot saved as '{output_path}'")
        plt.close()

    def calculate_enhanced_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        scored_recall = scored_questions['true_relevant'].sum() / max(scored_questions['expected_relevant'].sum(), 1)

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

    def print_enhanced_summary(self, enhanced_summary: Dict[str, Any]) -> None:
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

    def write_results_to_file(self, results: List[Dict[str, Any]], summary: Dict[str, float], enhanced_summary: Dict[str, Any], output_path: Path) -> None:
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

    def capture_comparison_data(self, query: Union[str, List[str]], retrieved_memories: List[Union[Dict[str, Any], Any, RetrievedMemory]], filtered_memories: List[Union[Dict[str, Any], Any, RetrievedMemory]]) -> None:
        comparison_data = {
            "query": query,
            "retrieved_memories_count": len(retrieved_memories),
            "filtered_memories_count": len(filtered_memories),
            "memories": self._format_memories(filtered_memories)
        }

        try:
            existing_data = self.file_manager.read_data()
            existing_data.append(comparison_data)
            self.file_manager.write_data(existing_data)
            self.logger.info(f"Comparison data appended to {self.file_manager.comparison_data_file}")
        except Exception as e:
            self.logger.error(f"Error capturing comparison data: {str(e)}")
            raise

    def _format_memories(self, memories: List[Union[Dict[str, Any], Any, RetrievedMemory]]) -> List[Dict[str, Any]]:
        formatted_memories = []
        for memory in memories:
            if isinstance(memory, dict):
                formatted_memory = memory.copy()
                formatted_memory['timestamp'] = str(formatted_memory.get('timestamp', ''))
            elif isinstance(memory, tuple):
                formatted_memory = {
                    "id": None,
                    "query": memory[0].content,
                    "response": memory[1].content,
                    "timestamp": memory[0].timestamp
                }
            else:
                formatted_memory = {
                    "id": getattr(memory, 'id', None),
                    "query": getattr(memory, 'query', ''),
                    "response": getattr(memory, 'response', ''),
                    "timestamp": str(getattr(memory, 'timestamp', '')),
                }
            formatted_memories.append(formatted_memory)
        return formatted_memories

    def _get_query_key(self, query: Union[str, List[str]]) -> str:
        if isinstance(query, list):
            return "\n".join(query)
        return query

    def output(self, benchmark_version: str) -> None:
        self.setup_logging(benchmark_version)

        reference_data_file = self.dev_bench_reference_data_path.joinpath(benchmark_version, "reference_data.json")

        if not reference_data_file.exists():
            raise FileNotFoundError(f"Reference data file for benchmark version {benchmark_version} not found at {reference_data_file}")

        reference_data = self.load_json_file(reference_data_file)
        input_data = self.file_manager.read_data()

        if not input_data:
            raise FileNotFoundError("No comparison data found. Please run the benchmark first.")

        self.results, self.summary = self.evaluate_memories(reference_data, input_data)
        self.enhanced_summary = self.calculate_enhanced_summary(self.results)

        timestamp = int(time.time())
        recall_score = self.summary['recall_score']

        output_dir = self.evaluation_outputs_path / f"evaluation_{benchmark_version}_{timestamp}"
        self.file_manager.ensure_directory_exists(output_dir)

        # Move the old comparison data to the output directory
        self.file_manager.clear_comparison_data(output_dir, timestamp)

        original_plot_path = output_dir / f"comparison_data_{benchmark_version}_{recall_score:.2f}_original.png"
        self.plot_original_results(self.results, self.summary, original_plot_path, benchmark_version)

        test_plot_path = output_dir / f"comparison_data_{benchmark_version}_{recall_score:.2f}_test_breakdown.png"
        self.plot_test_breakdown(self.results, test_plot_path)

        rf_plot_path = output_dir / f"comparison_data_{benchmark_version}_{recall_score:.2f}_retrieved_filtered.png"
        self.plot_retrieved_vs_filtered(self.results, rf_plot_path)

        self.print_enhanced_summary(self.enhanced_summary)

        results_path = output_dir / f"results_{benchmark_version}_{recall_score:.2f}.txt"
        self.write_results_to_file(self.results, self.summary, self.enhanced_summary, results_path)

        self.logger.info(f"\nEvaluation output directory: {output_dir}")
        self.logger.info(f"Log file: {self.logs_path / f'evaluation_{benchmark_version}.log'}")

def main():
    parser = argparse.ArgumentParser(description="Run RetrievalEvaluator for a specific benchmark version.")
    parser.add_argument("reference_data", help="Specify which reference data to compare against.")
    args = parser.parse_args()

    evaluator = RetrievalEvaluator()
    evaluator.output(args.reference_data)

if __name__ == "__main__":
    main()
