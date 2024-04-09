import click
import webbrowser
from reporting.generate import gather_results, generate_report


@click.command("detailed-report")
@click.argument("run_name", type=str)
@click.argument("agent_name", type=str)
@click.option("-o", "--output", type=str, required=False, default=None, help="Name of the resulting report, without extension.")
@click.option("-s", "--show", type=bool, required=False, default=True, help="Show the report in a new browser tab.")
def main(run_name: str, agent_name: str, output: str, show: bool):
    _main(run_name, agent_name, output, show)


def _main(run_name: str, agent_name: str, output: str, show: bool = True):
    benchmark_data, results = gather_results(run_name, agent_name)
    report_path = generate_report(results, output_name=output)
    if show:
        webbrowser.open_new_tab(report_path.as_uri())


if __name__ == "__main__":
    main()
