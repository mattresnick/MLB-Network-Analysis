Developer quickstart (VS Code + F5)

1) Press F5 and choose one of the MLB configurations (Full pipeline, Edges only, Rank only).
   - The first run creates a repo-local virtual environment at .venv and installs requirements.
   - The default config used is config/example_frequency_cv.json.

2) To change the config, either:
   - Edit .vscode/launch.json args, or
   - Open Run and Debug panel and override the argument, or
   - Duplicate the launch config with your preferred config path.

3) Manual setup (optional):
   - Run scripts/setup-venv.ps1 once to provision the venv.
   - Select Python interpreter: .venv/Scripts/python.exe in the VS Code status bar.

Outputs will be written to outputs/ by default (configurable in the JSON config).
