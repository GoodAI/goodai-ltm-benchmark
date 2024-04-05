import os
import json
import re

import yaml
from typing import List, Optional
from random import Random
from jinja2 import Environment, FileSystemLoader
from reporting.results import TestResult
from utils.files import gather_result_files, gather_runstats_files, make_config_path
from utils.constants import REPORT_TEMPLATES_DIR, MAIN_DIR, GOODAI_RED, GOODAI_GREEN, METRIC_NAMES, METRIC_ALT, \
    METRIC_UNITS, SPIDER_LABELS_OVERRIDE, REPORT_OUTPUT_DIR
from utils.data import load_b64
from utils.ui import display_float_or_int
from datetime import datetime
from pathlib import Path


def gather_results(run_name: str, agent_name: str):
    result_files = gather_result_files(run_name, agent_name)
    results = [TestResult.from_file(path) for path in result_files]
    benchmark_data_file = gather_runstats_files(run_name, agent_name)[0]
    with open(benchmark_data_file) as fd:
        benchmark_data = json.load(fd)
    return benchmark_data, results


def get_agent_color(agent_name: str) -> tuple[int, int, int]:
    r = Random(agent_name)
    return r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)


def formatted_log(result: TestResult) -> list[str]:
    line_color = {"Test": "659936", "Agent": "4249bc"}
    start_pattern = r"^((Test|Agent) (\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}\))?:)"
    task_log = []
    for line in result.task_log:
        m = re.match(start_pattern, line)
        assert m is not None, f"Unexpected log line: {line}"
        role = m.group(2)
        sender = m.group(0)
        msg = line.removeprefix(sender)
        task_log.append(f'<b><font color="#{line_color[role]}">{sender}</font></b>{msg}')
    return task_log


def arrange_data(results: List[TestResult]):

    run_name = results[0].run_name
    agent_name = results[0].agent_name
    max_score = achieved_score = 0
    info_gaps = list()
    data = dict()

    for res in results:
        assert res.run_name == run_name, "Can't create a detailed report of multiple runs."
        assert res.agent_name == agent_name, "Can't create a detailed report of multiple agents."

        if res.dataset_name not in data:
            data[res.dataset_name] = {
                "name": res.dataset_name,
                "description": res.description,
                "tests": [],
            }

        max_score += res.max_score
        achieved_score += res.score
        info_gaps.append(res.tokens)

        expected = [str(r) for r in res.expected_responses]
        actual = [str(r) for r in res.actual_responses]
        reasoning = [str(r) for r in res.reasoning]

        response_lens = {len(expected), len(actual), len(reasoning)}
        if len(response_lens) > 1:
            responses = (expected, actual, reasoning)
            responses = [tuple("\n".join(lines) for lines in responses)]
        else:
            responses = list(zip(expected, actual, reasoning))

        accuracy = res.score / res.max_score
        color = tuple(int(accuracy * GOODAI_GREEN[i] + (1 - accuracy) * GOODAI_RED[i]) for i in range(3))
        color = f"rgb{color}"

        test_dict = {
            "task_log": formatted_log(res),
            "responses": responses,
            "score": display_float_or_int(res.score),
            "max_score": display_float_or_int(res.max_score),
            "tokens": res.tokens,
            "characters": res.characters,
            "color": color,
        }
        data[res.dataset_name]["tests"].append(test_dict)

    with open(make_config_path(run_name)) as fd:
        config = yaml.safe_load(fd)
    args = config["datasets"]["args"]

    return dict(
        achieved_score=display_float_or_int(achieved_score),
        max_score=display_float_or_int(max_score),
        info_gap=0,  # TODO
        run_name=run_name,
        agent_name=agent_name,
        data_by_dataset=data,
        min_gap=min(info_gaps),
        max_gap=max(info_gaps),
        avg_gap=int(sum(info_gaps)/len(info_gaps)),
    )


def render_template(template_name: str, output_name: str = None, **kwargs) -> Path | str:
    file_loader = FileSystemLoader(REPORT_TEMPLATES_DIR)
    env = Environment(loader=file_loader)
    template = env.get_template(f"{template_name}.html")
    output = template.render(**kwargs)
    if output_name is None:
        return output
    path = REPORT_OUTPUT_DIR.joinpath(f"{output_name}.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(output)
    return path


def format_metric(value: float, metric_name: str) -> str:
    metric_name = metric_name.lower()
    if metric_name in ["accuracy", "verbosity", "score", "ltm"]:
        return str(int(value))
    return f"{value:.2f}"


def generate_report(results: List[TestResult], output_name: Optional[str] = None) -> Path:
    report_data = arrange_data(results)
    metrics = get_summary_data(report_data["run_name"], report_data["agent_name"])
    del metrics["score"]
    global_metrics = list()
    for key in sorted(metrics.keys()):
        global_metrics.append(dict(
            name=METRIC_NAMES[key],
            value=format_metric(metrics[key], key),
            alt=METRIC_ALT[key],
            units=METRIC_UNITS[key],
        ))
    if output_name is None:
        date = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        output_name = f"{date} - Detailed Report - {results[0].run_name} - {results[0].agent_name}"
    return render_template(
        template_name="detailed_report",
        output_name=output_name,
        logo_b64=load_b64(REPORT_TEMPLATES_DIR.joinpath("GoodAI_logo.png")),
        enumerate=enumerate,
        sorted=sorted,
        global_metrics=global_metrics,
        **arrange_data(results),
    )


def get_summary_data(run_name: str, agent_name: str):
    benchmark_data, results = gather_results(run_name, agent_name)
    score = max_score = accuracy = ltm_score = max_ltm_score = 0

    assert len(results) > 0, f"No results were found for run {run_name} and agent {agent_name}."

    for result in results:
        score += result.score
        max_score += result.max_score
        acc = result.score / result.max_score
        accuracy += acc
        ltm_score += result.tokens * acc
        max_ltm_score += result.tokens

    return dict(
        speed=len(results) / (benchmark_data["duration"] / 3600),
        cost=benchmark_data["agent_costs_usd"],
        verbosity=benchmark_data["agent_tokens"],
        score=score,
        accuracy=100 * accuracy / len(results),
        ltm=ltm_score,
    )


def generate_summary_report(
    run_name: str,
    agent_names: list[str],
    short_names: list[str] | None = None,
    output_name: str | None = None,
):
    metric_keys = sorted(METRIC_NAMES.keys())
    metric_names = [METRIC_NAMES[k] for k in metric_keys]
    data_by_agent = [dict() for _ in agent_names]
    data_by_metric = [dict() for _ in metric_names]
    summary_data_by_agent = {name: get_summary_data(run_name, name) for name in agent_names}

    # Organise data by agent
    # This data will be used in the spider chart
    for i, agent in enumerate(data_by_agent):
        agent_name = agent_names[i]
        agent["name"] = agent_name if short_names is None else short_names[i]
        r, g, b = get_agent_color(agent_name)
        agent_color = f"rgba({r}, {g}, {b}, 1)"
        agent_color_transparent = f"rgba({r}, {g}, {b}, 0.2)"
        agent["background_color"] = agent_color_transparent
        for k in ["border", "point_background", "point_hover_border"]:
            agent[k + "_color"] = agent_color
        agent["normalised_scores"] = list()
        agent["scores"] = list()
        for m in metric_keys:
            score = summary_data_by_agent[agent_name][m]
            agent["normalised_scores"].append(-score if m in SPIDER_LABELS_OVERRIDE else score)
            agent["scores"].append(score)

    # The spider plot is used to compare agents
    # Metrics are normalised to the interval [0.2, 1]
    # (A spider chart is hard to interpret when there are values right in the center)
    min_values = dict()
    max_values = dict()
    for i, k in enumerate(metric_keys):
        norm_scores = [agent["normalised_scores"][i] for agent in data_by_agent]
        min_values[k] = min(norm_scores)
        max_values[k] = max(norm_scores)
    for i, k in enumerate(metric_keys):
        for agent in data_by_agent:
            score = agent["normalised_scores"][i]
            score = (score - min_values[k]) / (max_values[k] - min_values[k] + 1e-8)
            agent["normalised_scores"][i] = 0.2 + 0.8 * score

    # Organise data by metric
    # This data will be used to create the individual bar plots
    for i, metric in enumerate(data_by_metric):
        metric["name"] = metric_names[i]
        metric["name_lower"] = metric_keys[i]
        metric["name_alt"] = METRIC_ALT[metric_keys[i]]
        scores_by_agent = list()
        for agent in data_by_agent:
            scores_by_agent.append(
                dict(
                    name=agent["name"],
                    background_color=agent["background_color"],
                    border_color=agent["border_color"],
                    score=agent["scores"][i],
                )
            )
        metric["scores_by_agent"] = scores_by_agent

    with open(make_config_path(run_name)) as fd:
        config = yaml.safe_load(fd)
    args = config["datasets"]["args"]
    info_gap = max(args["filler_tokens"], args["pre_question_filler"])

    # Embed the GoodAI logo and Chart.js into the HTML file.
    # This will avoid the need for an Internet connection to visualise the results.
    with open(REPORT_TEMPLATES_DIR.joinpath("chart.js")) as fd:
        chart_js = fd.read()

    if output_name is None:
        output_name = f"{datetime.now().strftime('%Y-%m-%d %H_%M_%S')} - Comparative Report - {run_name}"

    return render_template(
        template_name="comparative_report",
        output_name=output_name,
        spider_labels=[SPIDER_LABELS_OVERRIDE.get(k, METRIC_NAMES[k]) for k in metric_keys],
        data_by_agent=data_by_agent,
        data_by_metric=data_by_metric,
        max_spider_chart_width=30,
        max_bar_chart_width=70 // (len(metric_names) + 2),
        run_name=run_name,
        logo_b64=load_b64(REPORT_TEMPLATES_DIR.joinpath("GoodAI_logo.png")),
        chart_js=chart_js,
        info_gap=info_gap,
    )


def load_results_file(filename):
    full_file = "data" + os.sep + "results" + os.sep + filename
    results_list = []
    with open(full_file, "r", encoding="utf-8") as f:
        line = f.readline()
        while line != "":
            args = json.loads(line)
            results_list.append(TestResult(**args))
            line = f.readline()

    return results_list


if __name__ == "__main__":
    results = [
        TestResult(
            run_name="Run title",
            agent_name="Charlie",
            dataset_name="bar",
            description="foo",
            example_id="0",
            task_log=["A:", "B:", "C:"],
            expected_responses=["expected"],
            actual_responses=["actual"],
            reasoning=["reasoning"],
            score=0,
            max_score=1,
            tokens=1,
            characters=88,
        ),
        TestResult(
            run_name="Run title",
            agent_name="Charlie",
            dataset_name="bar",
            description="fasdas",
            example_id="1",
            task_log=["A:", "B:", "C:"],
            expected_responses=["expected"],
            actual_responses=["actual"],
            reasoning=["reasoning"],
            score=1,
            max_score=1,
            tokens=1,
            characters=88,
        ),
        TestResult(
            run_name="Run title",
            agent_name="Charlie",
            dataset_name="bar",
            description="foo",
            example_id="2",
            task_log=["A:", "B:", "C:"],
            expected_responses=["expected"],
            actual_responses=["actual"],
            reasoning=["reasoning"],
            score=1,
            max_score=5,
            tokens=1,
            characters=88,
        ),
    ]
    # results = load_results_file("colour")

    generate_report(results)
