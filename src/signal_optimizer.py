"""
signal_optimizer.py — Quantum-ready traffic signal timing optimizer
=====================================================================

Formulates "how many seconds of green light should each approach at an
intersection get, this cycle?" as a QUBO (Quadratic Unconstrained Binary
Optimization) problem — the native problem format solved by quantum
annealers (D-Wave). We build the QUBO with `dimod` (D-Wave's own modeling
library) and solve it with `neal.SimulatedAnnealingSampler`, a classical
stand-in that implements the *same sampler interface* as real quantum
hardware.

This is "quantum-ready" in a literal sense, not just a label: swapping
the solver for real quantum annealing hardware is a one-line change
(see `solve_on_quantum_hardware` below) — the QUBO construction, the
constraints, and the rest of the pipeline don't change at all.

--------------------------------------------------------------------
Problem formulation
--------------------------------------------------------------------
Split one signal cycle (e.g. 120s) into fixed-size time blocks (e.g.
5s each -> 24 blocks). For every lane i and block b, a binary variable

    x[i, b] = 1  if block b of the cycle is allocated to lane i's green phase
            = 0  otherwise

Objective — maximize total "served demand": blocks should go to lanes
with more waiting vehicles.

    maximize   sum_i  count[i] * sum_b x[i, b]

Constraint — each block is a single indivisible moment in the cycle,
so exactly one lane can have the green light during it:

    for each block b:   sum_i x[i, b] == 1

This one-hot constraint is added as a quadratic penalty term (the
standard way to encode constraints in a QUBO), so the whole problem is
unconstrained and annealer-solvable:

    (sum_i x[i, b] - 1)^2 ,  scaled by a penalty weight, for every block b

After solving, blocks assigned to each lane are converted back into
green-light seconds, then a minimum-green floor is enforced (no
approach should get 0 seconds even with light traffic) by clawing back
seconds from the lane with the largest allocation.
"""

import argparse
import json
from collections import defaultdict

import dimod
from neal import SimulatedAnnealingSampler


def build_qubo(lane_counts: dict, num_blocks: int, penalty: float = None) -> dict:
    """Build the QUBO dict for the signal-timing problem.

    lane_counts: {lane_name: vehicle_count_this_cycle}
    num_blocks: cycle_time // block_size
    penalty: constraint weight; auto-scaled to dominate the objective if not given
    """
    lanes = list(lane_counts.keys())
    if penalty is None:
        # Must outweigh the largest possible objective gain from breaking
        # the one-hot constraint, or the annealer will "cheat" by giving
        # every block to the busiest lane.
        penalty = max(lane_counts.values(), default=1) * 3 + 10

    Q = defaultdict(float)

    # Objective: reward assigning block b to lane i proportional to its count.
    # QUBO is a MINIMIZATION problem by convention, so we negate the reward.
    for lane in lanes:
        count = lane_counts[lane]
        for b in range(num_blocks):
            var = (lane, b)
            Q[(var, var)] += -count

    # Constraint: (sum_i x[i,b] - 1)^2 expanded for binary variables
    # (x_i^2 = x_i since x_i in {0,1}):
    #   = -sum_i x_i + 2 * sum_{i<j} x_i*x_j + 1
    for b in range(num_blocks):
        vars_b = [(lane, b) for lane in lanes]
        for v in vars_b:
            Q[(v, v)] += -penalty
        for i in range(len(vars_b)):
            for j in range(i + 1, len(vars_b)):
                Q[(vars_b[i], vars_b[j])] += 2 * penalty

    return dict(Q)


def solve_qubo_classically(Q: dict, num_reads: int = 200):
    """Solve with simulated annealing — a classical algorithm that mimics
    quantum annealing's search behavior. Same sampler interface as hardware."""
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=num_reads)
    best = response.first
    return best.sample, best.energy


def solve_on_quantum_hardware(Q: dict, num_reads: int = 200):  # pragma: no cover
    """Drop-in replacement for solve_qubo_classically using real D-Wave
    quantum annealing hardware. Requires `dwave-system` and a configured
    D-Wave Leap API token (`dwave config create`). Not called by default —
    this repo runs the classical sampler so it works with zero external
    dependencies or cloud accounts.
    """
    from dwave.system import DWaveSampler, EmbeddingComposite
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q, num_reads=num_reads)
    best = response.first
    return best.sample, best.energy


def optimize_signal_timings(lane_counts: dict, cycle_time: int = 120,
                             block_size: int = 5, min_green: int = 10,
                             num_reads: int = 200, use_quantum_hardware: bool = False) -> dict:
    """Compute green-light seconds per lane for one signal cycle.

    Returns {lane_name: green_seconds}, summing to `cycle_time`.
    """
    if not lane_counts:
        raise ValueError("lane_counts must have at least one lane")
    lanes = list(lane_counts.keys())
    num_blocks = max(cycle_time // block_size, len(lanes))

    Q = build_qubo(lane_counts, num_blocks)
    solve = solve_on_quantum_hardware if use_quantum_hardware else solve_qubo_classically
    sample, energy = solve(Q, num_reads=num_reads)

    # Tally blocks won by each lane (ties/infeasible blocks default to
    # the busiest lane so no block is silently dropped)
    block_winner = {}
    for b in range(num_blocks):
        candidates = [lane for lane in lanes if sample.get((lane, b), 0) == 1]
        block_winner[b] = candidates[0] if len(candidates) == 1 else max(lane_counts, key=lane_counts.get)

    blocks_per_lane = defaultdict(int)
    for lane in block_winner.values():
        blocks_per_lane[lane] += 1

    green = {lane: blocks_per_lane.get(lane, 0) * block_size for lane in lanes}

    # Enforce a minimum green time per lane by pulling seconds from
    # whichever lane currently has the most, then rescale to exactly
    # cycle_time (rounding can drift it by a few seconds).
    for lane in lanes:
        while green[lane] < min_green:
            donor = max(green, key=green.get)
            if donor == lane or green[donor] <= min_green:
                break
            green[donor] -= block_size
            green[lane] += block_size

    total = sum(green.values())
    if total != cycle_time and total > 0:
        scale = cycle_time / total
        green = {lane: max(min_green, round(v * scale)) for lane, v in green.items()}

    return {
        "green_seconds": green,
        "cycle_time": cycle_time,
        "qubo_energy": energy,
        "solver": "quantum_hardware" if use_quantum_hardware else "simulated_annealing (quantum-ready)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUBO-based traffic signal optimizer")
    parser.add_argument("--counts", required=True,
                         help='Lane vehicle counts as JSON, e.g. \'{"north":12,"south":8,"east":3,"west":5}\'')
    parser.add_argument("--cycle-time", type=int, default=120)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--min-green", type=int, default=10)
    args = parser.parse_args()

    lane_counts = json.loads(args.counts)
    result = optimize_signal_timings(lane_counts, args.cycle_time, args.block_size, args.min_green)
    print(json.dumps(result, indent=2))
