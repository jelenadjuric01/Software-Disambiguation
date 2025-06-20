# tests/test_cli.py

import subprocess
import sys
import os

def test_cli_help_and_process(tmp_path):
    # Force UTF-8 in both child and parent
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # 1) Help via module
    help_proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "sonad.cli", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    assert help_proc.returncode == 0
    assert "Usage" in help_proc.stdout

    # 2) Write the real input CSV
    input_csv = tmp_path / "input.csv"
    input_csv.write_text(
        'name,doi,paragraph\n'
        'Widoco,10.3390/buildings12101522,'
        '"Documentation and publication: Creating HTML documentation using WIDOCO. '
        'Multiple practical examples were elaborated on. '
        'An example dataset, instantiating the ontology, was openly published under the CC-BY 4.0 license"\n'
    )

    output_csv = tmp_path / "output.csv"

    # 3) Run the process command
    proc = subprocess.run(
        [
            sys.executable, "-X", "utf8", "-m", "sonad.cli", "process",
            "-i", str(input_csv),
            "-o", str(output_csv)
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"

    # 4) Verify output
    assert output_csv.exists(), "Output file wasn’t created"
    lines = output_csv.read_text().splitlines()
    assert len(lines) >= 2, "Expected at least header + one row"
