import click
import webbrowser
from reporting.generate import gather_results, generate_report


@click.command("detailed-report")
@click.argument("run_name", type=str)
@click.argument("agent_name", type=str)
@click.option("-o", "--output", type=str, required=False, default=None, help="Name of the resulting report, without extension.")
def main(run_name: str, agent_name: str, output: str):
    _main(run_name, agent_name, output)

def _main(run_name: str, agent_name: str, output: str):
    benchmark_data, results = gather_results(run_name, agent_name)
    report_path = generate_report(results, output_name=output)
    webbrowser.open_new_tab(report_path.as_uri())


if __name__ == "__main__":
    main()
