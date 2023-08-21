from pathlib import Path
import pytest
import subprocess

@pytest.mark.parametrize("example_filename", Path("examples").glob('*.py'))
def test_run_through_all_test(example_filename):    
    print(f'Running {example_filename}')
    subprocess.run(['python', example_filename], check=True)