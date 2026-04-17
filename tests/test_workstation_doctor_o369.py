"""Tests for scripts/workstation/doctor_o369.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "workstation" / "doctor_o369.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("doctor_o369", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# check_korali_install
# ---------------------------------------------------------------------------


def test_check_korali_install_pass(tmp_path, monkeypatch):
    module = _load_module()
    korali_dir = tmp_path / "_vega" / "korali" / "install"
    korali_dir.mkdir(parents=True)
    monkeypatch.setattr(module, "KORALI_INSTALL", korali_dir)
    level, msg = module.check_korali_install()
    assert level == "PASS"
    assert "found" in msg


def test_check_korali_install_fail(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "KORALI_INSTALL", tmp_path / "nonexistent")
    level, msg = module.check_korali_install()
    assert level == "FAIL"
    assert "missing" in msg


# ---------------------------------------------------------------------------
# check_korali_env_sh
# ---------------------------------------------------------------------------


def test_check_korali_env_sh_pass(tmp_path, monkeypatch):
    module = _load_module()
    env_sh = tmp_path / "_vega" / "korali" / "env.sh"
    env_sh.parent.mkdir(parents=True)
    env_sh.write_text("export PATH=...\n", encoding="utf-8")
    monkeypatch.setattr(module, "KORALI_ENV_SH", env_sh)
    level, msg = module.check_korali_env_sh()
    assert level == "PASS"


def test_check_korali_env_sh_fail(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "KORALI_ENV_SH", tmp_path / "missing" / "env.sh")
    level, msg = module.check_korali_env_sh()
    assert level == "FAIL"


# ---------------------------------------------------------------------------
# check_korali_import
# ---------------------------------------------------------------------------


def test_check_korali_import_resolves_inside_install(tmp_path, monkeypatch):
    module = _load_module()
    korali_dir = tmp_path / "_vega" / "korali" / "install"
    monkeypatch.setattr(module, "KORALI_INSTALL", korali_dir)
    korali_file = str(korali_dir / "lib" / "korali" / "__init__.py")

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = korali_file + "\n"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result):
        level, msg = module.check_korali_import(python_bin=sys.executable)

    assert level == "PASS"
    assert "_vega/korali/install" in msg


def test_check_korali_import_resolves_outside_install(tmp_path, monkeypatch):
    module = _load_module()
    korali_dir = tmp_path / "_vega" / "korali" / "install"
    monkeypatch.setattr(module, "KORALI_INSTALL", korali_dir)

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "/usr/lib/python3/dist-packages/korali/__init__.py\n"
    fake_result.stderr = ""

    with patch("subprocess.run", return_value=fake_result):
        level, msg = module.check_korali_import(python_bin=sys.executable)

    assert level == "WARN"
    assert "NOT under" in msg


def test_check_korali_import_fails_import(monkeypatch):
    module = _load_module()
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stdout = ""
    fake_result.stderr = "ModuleNotFoundError: No module named 'korali'\n"

    with patch("subprocess.run", return_value=fake_result):
        level, msg = module.check_korali_import(python_bin=sys.executable)

    assert level == "FAIL"
    assert "failed" in msg.lower()


def test_check_korali_import_python_not_found(monkeypatch):
    module = _load_module()
    with patch("subprocess.run", side_effect=FileNotFoundError):
        level, msg = module.check_korali_import(python_bin="/nonexistent/python")
    assert level == "FAIL"
    assert "not found" in msg


def test_check_korali_import_timeout(monkeypatch):
    module = _load_module()
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=30)):
        level, msg = module.check_korali_import(python_bin=sys.executable)
    assert level == "FAIL"
    assert "imeout" in msg


# ---------------------------------------------------------------------------
# check_mpirun
# ---------------------------------------------------------------------------


def test_check_mpirun_found(monkeypatch):
    module = _load_module()
    with patch("shutil.which", return_value="/usr/bin/mpirun"):
        level, msg = module.check_mpirun()
    assert level == "PASS"
    assert "/usr/bin/mpirun" in msg


def test_check_mpirun_missing(monkeypatch):
    module = _load_module()
    with patch("shutil.which", return_value=None):
        level, msg = module.check_mpirun()
    assert level == "FAIL"
    assert "not found" in msg


# ---------------------------------------------------------------------------
# check_free_ram
# ---------------------------------------------------------------------------


def test_check_free_ram_pass(tmp_path):
    module = _load_module()
    meminfo = tmp_path / "meminfo"
    # 4 GB free
    content = "MemTotal:       16000000 kB\nMemAvailable:    4194304 kB\n"
    meminfo.write_text(content, encoding="utf-8")
    level, msg = module.check_free_ram(meminfo_path=str(meminfo))
    assert level == "PASS"
    assert "GB" in msg


def test_check_free_ram_fail(tmp_path):
    module = _load_module()
    meminfo = tmp_path / "meminfo"
    # 512 MB free
    meminfo.write_text("MemAvailable:     524288 kB\n", encoding="utf-8")
    level, msg = module.check_free_ram(meminfo_path=str(meminfo))
    assert level == "FAIL"
    assert "below" in msg


def test_check_free_ram_oserror(tmp_path):
    module = _load_module()
    level, msg = module.check_free_ram(meminfo_path=str(tmp_path / "no_such_file"))
    assert level == "WARN"
    assert "unavailable" in msg


# ---------------------------------------------------------------------------
# run_checks integration
# ---------------------------------------------------------------------------


def test_run_checks_returns_zero_on_all_pass(monkeypatch, capsys):
    module = _load_module()

    monkeypatch.setattr(module, "check_korali_install", lambda: ("PASS", "ok"))
    monkeypatch.setattr(module, "check_korali_env_sh", lambda: ("PASS", "ok"))
    monkeypatch.setattr(module, "check_korali_import", lambda python_bin=None: ("PASS", "ok"))
    monkeypatch.setattr(module, "check_mpirun", lambda: ("PASS", "ok"))
    monkeypatch.setattr(module, "check_free_ram", lambda: ("PASS", "ok"))

    rc = module.run_checks()
    assert rc == 0
    out = capsys.readouterr().out
    assert "all checks passed" in out


def test_run_checks_returns_nonzero_on_fail(monkeypatch, capsys):
    module = _load_module()

    monkeypatch.setattr(module, "check_korali_install", lambda: ("FAIL", "missing"))
    monkeypatch.setattr(module, "check_korali_env_sh", lambda: ("PASS", "ok"))
    monkeypatch.setattr(module, "check_korali_import", lambda python_bin=None: ("PASS", "ok"))
    monkeypatch.setattr(module, "check_mpirun", lambda: ("PASS", "ok"))
    monkeypatch.setattr(module, "check_free_ram", lambda: ("PASS", "ok"))

    rc = module.run_checks()
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
