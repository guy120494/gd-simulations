"""
Thin entrypoint + compatibility layer.

- Running from terminal:
    python main.py --help
    python main.py --experiment 1

- Importing (backwards compatible with previous `from main import ...`):
    from main import experiment_1_arbitrary_neurons
"""


def _run_cli() -> None:
    from wide_network_experiments.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    _run_cli()
else:
    # Re-export API for notebooks/scripts that do `from main import *`
    from wide_network_experiments.api import *  # noqa: F401,F403

