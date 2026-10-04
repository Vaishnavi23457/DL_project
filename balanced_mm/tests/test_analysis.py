import os

from bml.analysis import collect_runs, lastk_val, make_report, split_runs, sweep_rows

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")


def test_collect_and_split_repo_runs():
    runs = collect_runs(RUNS)
    assert len(runs) >= 6                      # uni-modal baselines + 4 fusion runs + ...
    groups, unimodal = split_runs(runs, "synthetic")
    assert set(groups) == {"none", "opm", "ogm", "both"}
    assert any(r["config"].get("modulation") == "none" for r in groups["none"])
    assert len(unimodal) == 2                  # image-only and text-only


def test_sweep_rows_from_seed0():
    groups, _ = split_runs(collect_runs(RUNS), "synthetic")
    rows = sweep_rows(groups)
    assert len(rows) == 4
    for r in rows:
        assert r["n_seeds"] >= 1
        assert "±" in r["best"]
        assert 0.0 <= float(r["best"].split(" ± ")[0]) <= 1.0


def test_lastk_val():
    runs = collect_runs(RUNS)
    groups, _ = split_runs(runs, "synthetic")
    v = lastk_val(groups["none"][0], k=3)
    assert 0.0 <= v <= 1.0


def test_make_report_writes_artifacts(tmp_path):
    res = make_report(RUNS, str(tmp_path))
    md = tmp_path / "RESULTS.md"
    assert md.is_file()
    text = md.read_text()
    assert "mean ± std" in text and "Uni-modal baselines" in text
    assert (tmp_path / "acc_curves.png").is_file()
    assert (tmp_path / "rho_curves.png").is_file()
    assert (tmp_path / "bar_val_acc.png").is_file()
    assert res["rows"]
