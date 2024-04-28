from pathlib import Path
import pytest
import runpy


@pytest.mark.parametrize("example_filename", Path("examples").glob("*.py"))
def test_run_through_all_test(example_filename):
    print(f"Running {example_filename}")
    runpy.run_path(example_filename)
