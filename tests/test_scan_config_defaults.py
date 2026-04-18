from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_pipeline
from src.config import DEFAULT_SCAN_CONFIG, SCAN_FOLDER_NAME, ScanConfig
import src.process_scan_results as process_scan_results
import src.scan_alphac_pyele as scan_alphac_pyele


def test_run_pipeline_session_name_comes_from_default_scan_config() -> None:
    assert run_pipeline.get_session_dir_name() == DEFAULT_SCAN_CONFIG.session_dir_name()


def test_scan_cli_defaults_follow_central_scan_config() -> None:
    args = scan_alphac_pyele.get_args([])

    assert ScanConfig(
        startA=args.startA,
        stopA=args.stopA,
        stepA=args.stepA,
        startD=args.startD,
        stopD=args.stopD,
        stepD=args.stepD,
    ) == DEFAULT_SCAN_CONFIG


def test_process_defaults_to_configured_scan_directory() -> None:
    args = process_scan_results.get_parser([])

    assert args.dir == SCAN_FOLDER_NAME


def test_process_directory_name_fallback_uses_central_default_steps(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan_A1.00e-05-1.00e-04_D1.00e-05-2.20e-05"
    scan_dir.mkdir()

    scan_config = process_scan_results.resolve_scan_config(scan_dir)

    assert scan_config == ScanConfig(
        startA=1.0e-5,
        stopA=1.0e-4,
        stepA=DEFAULT_SCAN_CONFIG.stepA,
        startD=1.0e-5,
        stopD=2.2e-5,
        stepD=DEFAULT_SCAN_CONFIG.stepD,
    )


def test_process_falls_back_to_central_scan_config_when_metadata_is_missing(tmp_path: Path) -> None:
    scan_dir = tmp_path / "analysis_without_metadata"
    scan_dir.mkdir()

    assert process_scan_results.resolve_scan_config(scan_dir) == DEFAULT_SCAN_CONFIG
