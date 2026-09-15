from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVOICE_APP = (
    PROJECT_ROOT
    / "invoice_app.py"
)

runpy.run_path(
    str(INVOICE_APP),
    run_name="__main__",
)
