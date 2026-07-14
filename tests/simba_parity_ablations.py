#!/usr/bin/env python3
"""SIMBA-parity ablation sweep for the live GNS3 experiment, via dash_cli.py.

Reproduces, on the full TIGER stack, the exact experiment grid that was run
offline by smartville-controller/simba_experiments.py (the runs documented in
simba/REAL_DATA_CALIBRATION.md and simba/UNCERTAINTY_THRESHOLD_ABLATION.md):

  ablation modes x DM agents x seeds

with the SIMBA mode names, mapped onto TIGER's mutually-exclusive
epistemic-slot knobs in `_select_unknown_cluster_action`:

- "drl":                 all three actions available to the learned policy
                         (all scripted-CTI knobs left at the profile defaults).
- "no_epistemic":        intrusion_detection.no_epistemic_actions=True -- any
                         agent-chosen action 2 is remapped to block.
- "greedy_cti":          intrusion_detection.greedy_cti=True -- buy forced
                         whenever an unbought G2 is on sale and affordable.
- "fixed_threshold_cti": intrusion_detection.fixed_threshold_cti=True -- buy
                         forced whenever the cluster's anomaly confidence
                         exceeds cti_confidence_threshold (and the CTI is on
                         sale/affordable); expands into ONE RUN PER value in
                         --cti-confidence-thresholds, i.e. SIMBA's threshold
                         sweep. NB: under the dista_tiger profile the cluster
                         confidence is the 'baseline' strategy = the cluster
                         members' mean anomaly probability, in (0, 1) -- so
                         thresholds live in (0, 1) here, unlike SIMBA's
                         dist-ratio scale (where they sat around 1.5).
- "oracle":              SIMBA's oracle-buyer upper bound: greedy_cti=True
                         LAYERED with hard_g2s=--hard-g2s (default: every
                         malicious G2), so greed is forced to buy only the
                         worthwhile benign zero-days. Mirrors
                         `simba_offline.py --mode greedy_cti --hard-g2s ...`.

The DM-agent axis is orthogonal (SIMBA's `--agents dqn ddqn dueling_ddqn`):
TIGER's ValueLearningAgent covers the same three learners as DQN, DDQN and
DuelingDDQN -- same hyperparameters, only the TD target / Q-head change.
Default seeds are 6 and 1, SIMBA's calibration-study seeds.

Everything else (health polling, applied seed/agent verification, fail-loud
aborts, seeds as the outermost loop) follows tests/seeded_agents.py; the
default --profile is dista_tiger, which is the SIMBA-parity configuration.

Example (the full offline comparison, live):

    python simba_parity_ablations.py \
        --ablation-modes drl no_epistemic greedy_cti \
        --agents DQN DDQN DuelingDDQN \
        --seeds 6 1 --run-duration-seconds 3600

    # the reviewer's threshold sweep + the oracle bound, DQN only:
    python simba_parity_ablations.py \
        --ablation-modes fixed_threshold_cti oracle --agents DQN --seeds 6 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# SIMBA's agent axis (simba/agent.py dqn | ddqn | dueling_ddqn), in TIGER's
# ValueLearningAgent naming.
DEFAULT_AGENTS = ["DQN", "DDQN", "DuelingDDQN"]

# SIMBA's calibration-study seeds (REAL_DATA_CALIBRATION.md ran 6 and 1).
DEFAULT_SEEDS = [6, 1]

DEFAULT_HEALTH_POLL_INTERVAL_SECONDS = 60

# Threshold sweep for fixed_threshold_cti. Under the dista_tiger profile the
# cluster confidence compared against the threshold is the 'baseline'
# confidence strategy: the cluster members' mean anomaly probability, in
# (0, 1). Values <= the profile's ad_threshold (0.5) buy essentially every
# unknown cluster (degenerating toward greedy_cti); values near 1 buy almost
# none (degenerating toward no_epistemic) -- same trade-off SIMBA's
# UNCERTAINTY_THRESHOLD_ABLATION.md sweeps on its dist-ratio scale.
DEFAULT_CTI_CONFIDENCE_THRESHOLDS = [0.6, 0.75, 0.9]

# SIMBA's oracle-buyer blocklist: all the not-worth-buying (malicious) G2s,
# leaving greedy free to buy only doorlock/echo.
DEFAULT_HARD_G2S = ["mirai", "gafgyt", "hajime", "h_scan", "muhstik"]

ABLATION_MODES = ["drl", "no_epistemic", "greedy_cti", "fixed_threshold_cti", "oracle"]


def expand_run_configs(modes: list[str], thresholds: list[float]) -> list[tuple[str, float | None]]:
    """Expand fixed_threshold_cti into one config per threshold; every other
    mode is a single config with threshold None."""
    configs: list[tuple[str, float | None]] = []
    for mode in modes:
        if mode == "fixed_threshold_cti":
            configs.extend((mode, t) for t in thresholds)
        else:
            configs.append((mode, None))
    return configs


def ablation_overrides(mode: str, threshold: float | None, hard_g2s: list[str]) -> dict[str, str]:
    if mode == "drl":
        return {}
    if mode == "no_epistemic":
        return {"intrusion_detection.no_epistemic_actions": "true"}
    if mode == "greedy_cti":
        return {"intrusion_detection.greedy_cti": "true"}
    if mode == "fixed_threshold_cti":
        return {"intrusion_detection.fixed_threshold_cti": "true",
                "intrusion_detection.cti_confidence_threshold": str(threshold)}
    if mode == "oracle":
        # greedy buying restricted to the classes worth buying: hard_g2s is an
        # orthogonal FILTER layered on greedy_cti (see default.yaml's hard_g2s
        # doc), exactly SIMBA's `greedy_cti --hard-g2s ...` oracle baseline.
        return {"intrusion_detection.greedy_cti": "true",
                "intrusion_detection.hard_g2s": "[" + ", ".join(hard_g2s) + "]"}
    raise ValueError(f"Unknown ablation mode: {mode!r}")


def wb_run_name(mode: str, threshold: float | None, agent: str, seed: int) -> str:
    """SIMBA's run naming: <mode>[_<agent>]_seed<seed>, with the threshold
    baked into the fixed_threshold_cti mode label (fixed_thr0.75_...). The
    plain DQN keeps the bare mode name, exactly like SIMBA's default `dqn`
    learner keeps `<mode>_seed<seed>`."""
    base = f"fixed_thr{threshold:g}" if mode == "fixed_threshold_cti" else mode
    if agent != "DQN":
        base = f"{base}_{agent.lower()}"
    return f"{base}_seed{seed}"


def fail_loudly(message: str) -> None:
    """
    Abort the whole sweep immediately and unmissably, instead of silently
    skipping a broken run or sleeping out a multi-hour duration that's
    already known to be invalid.
    """
    banner = "!" * 78
    print(f"\n{banner}\n[ABORT] {message}\n{banner}\n", file=sys.stderr, flush=True)
    sys.exit(1)


def run_step(script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    print(f"\n[step] Running: dash_cli.py {' '.join(cmd[2:])}\n", flush=True)
    subprocess.run(cmd, check=True)


def run_step_json(script: Path, args: list[str]) -> dict[str, Any]:
    """
    Like run_step, but via --raw-json so the response body can be inspected
    programmatically. Used for the calls whose actual effect (not just
    "the HTTP request didn't error") needs verifying: did the controller
    apply the seed/agent we asked for, and is it still alive.
    """
    cmd = [sys.executable, str(script), "--raw-json", *args]
    print(f"\n[step] Running: dash_cli.py --raw-json {' '.join(args)}\n", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)

    if result.returncode != 0:
        fail_loudly(f"dash_cli.py {' '.join(args)} exited with code {result.returncode}")

    try:
        last_line = result.stdout.strip().splitlines()[-1]
        parsed = json.loads(last_line)
        return parsed["body"]
    except (json.JSONDecodeError, IndexError, KeyError) as exc:
        fail_loudly(f"dash_cli.py {' '.join(args)} produced no parseable --raw-json output: {exc}")
        raise AssertionError("unreachable")  # for type-checkers; fail_loudly never returns


def sleep_with_spinner(seconds: int, label: str) -> None:
    spinner = "|/-\\"
    start = time.monotonic()
    tick = 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= seconds:
            break
        remaining = max(0, int(seconds - elapsed))
        frame = spinner[tick % len(spinner)]
        sys.stdout.write(f"\r[wait] {label} {frame} ({remaining}s remaining)")
        sys.stdout.flush()
        tick += 1
        time.sleep(0.1)
    sys.stdout.write(f"\r[done] {label} complete.{' ' * 20}\n")
    sys.stdout.flush()


def wait_with_health_checks(
    dash_cli_path: Path,
    total_seconds: int,
    run_name: str,
    poll_interval_seconds: int,
) -> None:
    """
    Sleeps out total_seconds in poll_interval_seconds chunks, polling
    /controller_health between chunks. A daemon thread crashing inside the
    controller leaves the process up and answering requests, so without this
    a dead run would otherwise look identical to a healthy one until the
    full multi-hour duration had already been wasted.
    """
    elapsed = 0.0
    while elapsed < total_seconds:
        chunk = min(poll_interval_seconds, total_seconds - elapsed)
        sleep_with_spinner(int(round(chunk)), f"{run_name} running ({int(elapsed + chunk)}/{total_seconds}s)...")
        elapsed += chunk

        body = run_step_json(dash_cli_path, ["check-controller-health"])
        if body.get("status") == "crashed":
            crash_info = body.get("crash_info", {})
            fail_loudly(
                f"{run_name}: controller's inference loop crashed mid-run at check "
                f"#{crash_info.get('check_count')} after {int(elapsed)}s: {crash_info.get('error')}"
            )
        elif body.get("status") != "ok":
            fail_loudly(f"{run_name}: unexpected controller health status: {body}")


def run_one(
    dash_cli_path: Path,
    profile: str,
    agent: str,
    mode: str,
    threshold: float | None,
    overrides: dict[str, str],
    seed: int,
    run_duration_seconds: int,
    group_name: str,
    health_poll_interval_seconds: int,
) -> None:
    run_name = wb_run_name(mode, threshold, agent, seed)
    print(f"[step] Setting up {run_name}...", flush=True)

    # Reload the profile fresh every run so a previous run's ablation knobs
    # (e.g. greedy_cti, hard_g2s) can never leak into the next one.
    run_step(dash_cli_path, ["--profile", profile, "init-config"])
    run_step(dash_cli_path, ["set", "intrusion_detection.agent", agent])
    run_step(dash_cli_path, ["set", "intrusion_detection.seed", str(seed)])
    for key, value in overrides.items():
        run_step(dash_cli_path, ["set", key, value])
    run_step(dash_cli_path, ["set", "wandb.wb_run_name", run_name])
    run_step(dash_cli_path, ["set", "wandb.wb_group_name", group_name])

    print(f"[step] Starting {run_name} and verifying applied config...", flush=True)
    body = run_step_json(dash_cli_path, ["start-experiment"])

    if body.get("status_code") != 200:
        fail_loudly(f"{run_name}: controller rejected /initialize: {body.get('msg')}")
    if str(body.get("applied_seed")) != str(seed):
        fail_loudly(
            f"{run_name}: requested seed {seed!r} but controller reports applied_seed="
            f"{body.get('applied_seed')!r} -- the seed was NOT transmitted/applied correctly."
        )
    if body.get("applied_agent") != agent:
        fail_loudly(
            f"{run_name}: requested agent {agent!r} but controller reports applied_agent="
            f"{body.get('applied_agent')!r}."
        )
    print(f"[ok] Confirmed controller applied seed={body['applied_seed']} agent={body['applied_agent']}", flush=True)

    run_step(dash_cli_path, ["start-traffic"])

    wait_with_health_checks(dash_cli_path, run_duration_seconds, run_name, health_poll_interval_seconds)

    print(f"[step] Stopping {run_name}...", flush=True)
    run_step(dash_cli_path, ["stop-experiment"])
    run_step(dash_cli_path, ["stop-traffic"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run SIMBA's offline experiment grid (ablation modes x DM agents x "
            "seeds) on the live GNS3 TIGER stack, under the SIMBA-parity "
            "dista_tiger profile."
        )
    )
    parser.add_argument(
        "--ablation-modes",
        nargs="+",
        default=["drl", "no_epistemic", "greedy_cti"],
        choices=ABLATION_MODES,
        help=(
            "SIMBA ablation modes to sweep (default: drl no_epistemic "
            "greedy_cti -- the REAL_DATA_CALIBRATION comparison). "
            "'fixed_threshold_cti' expands into one run per value of "
            "--cti-confidence-thresholds; 'oracle' is greedy_cti restricted by "
            "--hard-g2s to the worthwhile buys."
        ),
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=DEFAULT_AGENTS,
        choices=["DQN", "DDQN", "DuelingDQN", "DuelingDDQN"],
        help=(
            f"DM value-learners to sweep (default: {DEFAULT_AGENTS} = SIMBA's "
            "dqn/ddqn/dueling_ddqn axis). Orthogonal to the ablation modes: the "
            "modes gate the epistemic slot, the agent gates how Q-values are "
            "learned."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=(
            f"Seeds to run per (mode, agent) config (default: {DEFAULT_SEEDS}, "
            "SIMBA's calibration-study seeds). Seeds are the outermost loop, so "
            "an interrupted sweep leaves complete per-seed config sets."
        ),
    )
    parser.add_argument(
        "--cti-confidence-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_CTI_CONFIDENCE_THRESHOLDS,
        help=(
            "Threshold sweep for the fixed_threshold_cti mode (default: "
            f"{DEFAULT_CTI_CONFIDENCE_THRESHOLDS}). Compared against the "
            "cluster's mean anomaly probability under the dista_tiger profile's "
            "'baseline' confidence strategy, so values live in (0, 1)."
        ),
    )
    parser.add_argument(
        "--hard-g2s",
        nargs="+",
        default=DEFAULT_HARD_G2S,
        help=(
            "CTI-purchase blocklist used by the 'oracle' mode (default: "
            f"{DEFAULT_HARD_G2S} = every malicious G2, leaving greedy to buy "
            "only the benign zero-days)."
        ),
    )
    parser.add_argument(
        "--profile",
        default="dista_tiger",
        help=(
            "config/overrides/<profile>.yaml to (re)load before each run "
            "(default: dista_tiger, the SIMBA-parity configuration)."
        ),
    )
    parser.add_argument(
        "--wandb-group-name",
        default="simba-parity",
        help="W&B group name shared by all runs in this sweep (default: simba-parity).",
    )
    parser.add_argument(
        "--initial-delay-seconds",
        type=int,
        default=3,
        help="Delay before starting the sweep (default: 3 seconds).",
    )
    parser.add_argument(
        "--run-duration-seconds",
        type=int,
        default=120 * 60,
        help="Duration of each individual run (default: 120 minutes).",
    )
    parser.add_argument(
        "--health-poll-interval-seconds",
        type=int,
        default=DEFAULT_HEALTH_POLL_INTERVAL_SECONDS,
        help=(
            "How often to poll /controller_health during a run (default: "
            f"{DEFAULT_HEALTH_POLL_INTERVAL_SECONDS}s). A crash aborts the whole sweep "
            "at the next poll instead of after the full run duration."
        ),
    )
    parser.add_argument(
        "--dash-cli-path",
        type=Path,
        default=Path(__file__).resolve().parent / "dash_cli.py",
        help="Path to dash_cli.py.",
    )

    args = parser.parse_args()

    dash_cli_path = args.dash_cli_path.resolve()
    if not dash_cli_path.exists():
        print(f"dash_cli.py not found: {dash_cli_path}")
        return 1

    run_configs = expand_run_configs(args.ablation_modes, args.cti_confidence_thresholds)

    if args.initial_delay_seconds > 0:
        sleep_with_spinner(args.initial_delay_seconds, "Sleeping before starting sweep...")

    print("[step] Stopping any previous run...", flush=True)
    run_step(dash_cli_path, ["stop-experiment"])
    run_step(dash_cli_path, ["stop-traffic"])

    total_runs = len(args.seeds) * len(args.agents) * len(run_configs)
    run_idx = 0
    for seed in args.seeds:
        for agent in args.agents:
            for mode, threshold in run_configs:
                run_idx += 1
                overrides = ablation_overrides(mode, threshold, args.hard_g2s)
                thr_msg = f" threshold={threshold:g}" if threshold is not None else ""
                print(
                    f"\n===== Run {run_idx}/{total_runs}: "
                    f"seed={seed} agent={agent} mode={mode}{thr_msg} =====",
                    flush=True,
                )
                run_one(
                    dash_cli_path=dash_cli_path,
                    profile=args.profile,
                    agent=agent,
                    mode=mode,
                    threshold=threshold,
                    overrides=overrides,
                    seed=seed,
                    run_duration_seconds=args.run_duration_seconds,
                    group_name=args.wandb_group_name,
                    health_poll_interval_seconds=args.health_poll_interval_seconds,
                )

    print("[done] SIMBA-parity ablation sweep completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
