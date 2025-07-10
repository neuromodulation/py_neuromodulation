from pathlib import Path
import pytest
import runpy


@pytest.mark.parametrize("example_filename", Path("examples").glob("*.py"))
@pytest.mark.no_parallel
def test_run_through_all_test(example_filename):
    print(f"Running {example_filename}")
    if "plot_8_cebra_example.py" not in example_filename.name:
        runpy.run_path(example_filename)
