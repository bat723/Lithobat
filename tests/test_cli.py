"""
The command-line surface: every command registers, parses, and runs.

The commands are thin — each is a script's body moved behind a parser — so
what is worth pinning is the wiring: that the parser lists them all, that
each one's flags parse, and that the cheap ones run end to end and leave
their files where ``--output`` said.
"""

from __future__ import annotations

import pytest

from litho_sim.cli import build_parser, main

COMMANDS = ("demo", "bossung", "window", "opc", "multipatterning", "vector", "device",
            "stochastic")


def test_every_command_is_registered():
    parser = build_parser()
    subs = next(a for a in parser._actions if a.dest == "command")
    assert tuple(subs.choices) == COMMANDS


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_help(command, capsys):
    with pytest.raises(SystemExit) as exit_:
        build_parser().parse_args([command, "--help"])
    assert exit_.value.code == 0
    assert command in capsys.readouterr().out


@pytest.mark.parametrize("device", ("gaa", "nfet"))
def test_device_subcommands_carry_their_flows_flags(device):
    args = build_parser().parse_args(["device", device, "--calibrate"])
    assert args.device == device and args.calibrate
    assert callable(args.func)


def test_shared_flags_are_spelt_the_same_everywhere():
    parser = build_parser()
    for command in ("demo", "bossung", "window", "opc"):
        args = parser.parse_args([command, "--node", "EUV", "--no-show"])
        assert args.node == "EUV" and args.no_show


def test_demo_writes_its_figures(tmp_path):
    main(["demo", "--no-show", "--no-3d", "--output", str(tmp_path)])
    assert (tmp_path / "aerial_image.png").is_file()
    assert (tmp_path / "resist_profile.png").is_file()
    assert not (tmp_path / "resist_profile_3d.png").exists()


def test_window_writes_data_and_plots(tmp_path):
    main(["window", "--no-show", "--n-doses", "3", "--n-defoci", "5",
          "--output", str(tmp_path)])
    assert (tmp_path / "process_window_data.csv").is_file()
    assert (tmp_path / "bossung_curves.png").is_file()
    assert (tmp_path / "cd_heatmap.png").is_file()


def test_opc_runs_on_a_small_grid(tmp_path):
    out = tmp_path / "opc.png"
    main(["opc", "--pixels", "64", "--iterations", "1", "--no-show", "--output", str(out)])
    assert out.is_file()


def test_a_failing_command_exits_non_zero(monkeypatch):
    """A command's integer status becomes the process status."""
    from litho_sim.cli import basic

    monkeypatch.setattr(basic, "run_demo", lambda args: 3)
    with pytest.raises(SystemExit) as exit_:
        main(["demo", "--no-show", "--no-3d"])
    assert exit_.value.code == 3


@pytest.mark.slow
def test_device_nfet_builds_and_passes_its_checks(capsys):
    main(["device", "nfet"])
    assert "ALL CHECKS PASS" in capsys.readouterr().out
