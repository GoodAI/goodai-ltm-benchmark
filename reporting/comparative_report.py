import click
import webbrowser
from pathlib import Path
from utils.ui import ask_yesno
from utils.files import gather_runstats_files
from reporting.generate import generate_summary_report


@click.command("comparative-report")
@click.argument("run_name", type=str)
@click.option("-a", "--agents", type=str, multiple=True, required=False, help="List of agents (at least two elements)")
@click.option("-l", "--labels", type=str, multiple=True, required=False, help="Labels to use for the agents given.")
@click.option("-o", "--output", type=str, required=False, help="Name of the resulting report, without extension.")
def main(run_name: str, agents: list[str] | None, labels: list[str] | None, output: str | None):

    if agents is None or len(agents) < 2:
        labels = None  # Invalidate any provided label
        print("This report can only be generated for two agents or more.")
        summary_files = gather_runstats_files(run_name)
        agent_names = sorted(Path(path).parent.name for path in summary_files)
        if len(agent_names) < 2:
            print("Cannot generate the report")
            return
        print("Select two or more agents from the following list:")
        for i, name in enumerate(agent_names):
            print(f"{i}: {name}")
        agents = []
        while len(agents) < 2:
            print("Enter at least two indices, separated by commas. E.g. 1,2,4")
            sel = input("Selection: ")
            try:
                agents = sorted(agent_names[s] for s in [int(s.strip()) for s in sel.split(",")])
            except:
                pass

    if labels is None:
        if ask_yesno(question="Do you wish to provide labels for the selected agents?", default_yes=False):
            print("Please enter the labels one by one.")
            labels = list()
            for name in agents:
                agent_label = input(f"Label for {name}: ")
                labels.append(agent_label.strip())

    report_path = generate_summary_report(
        run_name=run_name,
        agent_names=agents,
        short_names=labels,
        output_name=output,
    )
    webbrowser.open_new_tab(report_path.as_uri())


if __name__ == "__main__":
    main()
