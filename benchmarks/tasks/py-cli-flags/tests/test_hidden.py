"""Hidden evaluation test for CLI flag parsing."""

import subprocess


def test_cli_name_flag():
    out = subprocess.run(
        ["python", "cli.py", "--name", "Ada"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "Hello, Ada!" in out.stdout


if __name__ == "__main__":
    test_cli_name_flag()
    print("ok")
