from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.elegant_extractor import ElegantDataExtractor


def test_elegant_extractor_reads_horizontal_twiss_and_dispersion(monkeypatch, tmp_path: Path) -> None:
    twiss_file = tmp_path / "sample_final.twi"
    param_file = tmp_path / "sample_check.param"
    twiss_file.write_text("dummy", encoding="utf-8")
    param_file.write_text("dummy", encoding="utf-8")

    responses = {
        f'sdds2stream "{twiss_file.as_posix()}" -parameter=pCentral': "1230.0",
        f'sdds2stream "{twiss_file.as_posix()}" -parameter=U0': "9100.0",
        f'sddsprocess "{twiss_file.as_posix()}" -pipe=out -process=s,max,Length | sdds2stream -pipe=in -parameter=Length': "48.0",
        f'sddsprocess "{twiss_file.as_posix()}" -pipe=out -match=col,ElementName=WISLANDP | sdds2stream -pipe=in -column=betax': "7.08",
        f'sddsprocess "{twiss_file.as_posix()}" -pipe=out -match=col,ElementName=WISLANDP | sdds2stream -pipe=in -column=alphax': "1.25",
        f'sddsprocess "{twiss_file.as_posix()}" -pipe=out -match=col,ElementName=WISLANDP | sdds2stream -pipe=in -column=etax': "0.345",
        f'sddsprocess "{twiss_file.as_posix()}" -pipe=out -match=col,ElementName=WISLANDP | sdds2stream -pipe=in -column=etaxp': "-0.012",
        f'sdds2stream "{twiss_file.as_posix()}" -parameter=ex0': "1.9e-7",
        f'sddsprocess "{param_file.as_posix()}" -pipe=out -match=col,ElementName=RF1 -match=col,ElementParameter=VOLT | sdds2stream -pipe=in -column=ParameterValue': "500000.0",
        f'sddsprocess "{param_file.as_posix()}" -pipe=out -match=col,ElementName=RF1 -match=col,ElementParameter=FREQ | sdds2stream -pipe=in -column=ParameterValue': "500000000.0",
        f'sdds2stream "{twiss_file.as_posix()}" -parameter=alphac': "1.0e-4",
        f'sdds2stream "{twiss_file.as_posix()}" -parameter=alphac2': "6.15e-2",
    }

    def fake_run_cmd(self: ElegantDataExtractor, cmd: str) -> str | None:
        return responses.get(cmd)

    monkeypatch.setattr(ElegantDataExtractor, "_run_cmd", fake_run_cmd)

    extractor = ElegantDataExtractor(
        twiss_file=twiss_file,
        param_file=param_file,
        rf_element_name="RF1",
        watch_element_name="WISLANDP",
    )

    assert extractor.beta_x == 7.08
    assert extractor.alpha_x == 1.25
    assert extractor.eta_x == 0.345
    assert extractor.eta_xp == -0.012
