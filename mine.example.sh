#!/usr/bin/env bash
#
# Example jackpotminer configuration.
#
# jackpotminer is configured entirely through command-line flags — there is no
# native config-file parser. This script is the recommended way to keep your
# settings: copy it, edit the values, and run it.
#
#   cargo build --release            # build the miner first
#   cp mine.example.sh mine.sh
#   $EDITOR mine.sh                  # set your pool + wallet, tweak the rest
#   ./mine.sh
#
# Run `./target/release/jackpotminer --help` for the authoritative list of options.

set -uo pipefail

BIN="./target/release/jackpotminer"

# ── Pool ──────────────────────────────────────────────────────────────────────
# Pool address. Accepts stratum+tcp://host:port, tcp://host:port, or host:port.
URL="stratum+tcp://zcl.pool.example:3032"

# Your payout address / worker login (often "address.workername").
WALLET="t1YourZClassicAddressHere.rig1"

# ── Payout mode (set at most ONE of SOLO / JACKPOT_PERCENT / NO_JACKPOT) ───────
PASS="jackpot"          # fallback when none of the three below is set
SOLO=false              # solo mining
JACKPOT_PERCENT=""      # 3-100 -> "jackpot.<n>" participation percent, e.g. 50
NO_JACKPOT=false        # PPLNS (opt out of the jackpot)

# ── GPU backend ───────────────────────────────────────────────────────────────
BACKEND="cuda"          # "cuda" (NVIDIA, fastest here) or "opencl"
FORCE_OPENCL=false      # force OpenCL, disabling CUDA (overrides BACKEND)
DEVICES="all"           # "all", or a comma list e.g. "0,1"

# ── GPU tuning (NVIDIA/CUDA; clock & power changes need root) ──────────────────
NO_GPU_TUNE=false       # skip the max-performance clock/power setup
AUTO_TUNE=false         # sweep core clock at startup for the fastest stable rate
POWER_LIMIT=""          # watts, e.g. 250 (efficiency: lower trades Sol/s for Sol/W)
GPU_CLOCK=""            # lock SM/core clock, MHz
MEM_CLOCK=""            # lock memory clock, MHz
GPU_CLOCK_OFFSET=""     # signed core V/F offset MHz, e.g. 200 or -150
MEM_CLOCK_OFFSET=""     # signed memory V/F offset MHz
UNLOCK_CONTROLS=false   # allow the dashboard's live clock/power keys (z/x/c/v/b/n)

# ── CPU mining (optional; runs alongside the GPU backend) ──────────────────────
USE_CPU_BACKEND=false   # --cpu : use the CPU as the *primary* solver (no GPU)
CPU_MINING=false        # start the CPU core rows enabled at launch
CPU_CORES=""            # which cores, e.g. "0-7" or "0,2,4,6" (default: all)
CPU_GROUP_SIZE=4        # cores per CPU mining row (1 = one solve per core)
CPU_CLAMP=32            # collision-bucket clamp (0 = exact solver, can OOM)

# ── Misc ──────────────────────────────────────────────────────────────────────
THREADS=""              # CPU threads for the solver (default: all cores)
JOB_TIMEOUT=300         # pause if no new job arrives within N seconds (0 = off)
NO_TUI=false            # disable the live dashboard; print periodic log lines

# ──────────────────────────────────────────────────────────────────────────────
# Build the argument list from the settings above (no need to edit below here).
args=(--url "$URL" --user "$WALLET" --backend "$BACKEND" --devices "$DEVICES"
      --cpu-group-size "$CPU_GROUP_SIZE" --cpu-clamp "$CPU_CLAMP"
      --job-timeout "$JOB_TIMEOUT")

# Payout mode (mutually exclusive)
if [ "$SOLO" = true ]; then
  args+=(--solo)
elif [ -n "$JACKPOT_PERCENT" ]; then
  args+=(--jackpot "$JACKPOT_PERCENT")
elif [ "$NO_JACKPOT" = true ]; then
  args+=(--no-jackpot)
else
  args+=(--pass "$PASS")
fi

# Backend selection
[ "$FORCE_OPENCL" = true ]   && args+=(--force-opencl)
[ "$USE_CPU_BACKEND" = true ] && args+=(--cpu)

# GPU tuning
[ "$NO_GPU_TUNE" = true ]    && args+=(--no-gpu-tune)
[ "$AUTO_TUNE" = true ]      && args+=(--auto-tune)
[ "$UNLOCK_CONTROLS" = true ] && args+=(--unlock-controls)
[ -n "$POWER_LIMIT" ]        && args+=(--power-limit "$POWER_LIMIT")
[ -n "$GPU_CLOCK" ]          && args+=(--gpu-clock "$GPU_CLOCK")
[ -n "$MEM_CLOCK" ]          && args+=(--mem-clock "$MEM_CLOCK")
[ -n "$GPU_CLOCK_OFFSET" ]   && args+=(--gpu-clock-offset "$GPU_CLOCK_OFFSET")
[ -n "$MEM_CLOCK_OFFSET" ]   && args+=(--mem-clock-offset "$MEM_CLOCK_OFFSET")

# CPU mining
[ "$CPU_MINING" = true ]     && args+=(--cpu-mining)
[ -n "$CPU_CORES" ]          && args+=(--cpu-cores "$CPU_CORES")

# Misc
[ -n "$THREADS" ]            && args+=(--threads "$THREADS")
[ "$NO_TUI" = true ]         && args+=(--no-tui)

echo "+ $BIN ${args[*]}"
exec "$BIN" "${args[@]}"

# ── Common setups (copy values into the section above) ────────────────────────
#  • All GPUs, CUDA, default jackpot:        BACKEND=cuda  DEVICES=all
#  • Two specific GPUs on OpenCL:            FORCE_OPENCL=true  DEVICES="0,1"
#  • Efficiency (undervolt-ish):             POWER_LIMIT=240  GPU_CLOCK_OFFSET=150  (needs root)
#  • GPU + spare CPU cores mining too:       CPU_MINING=true  CPU_CORES="0-7"  CPU_GROUP_SIZE=4
#  • CPU-only rig:                           USE_CPU_BACKEND=true  CPU_MINING=true
#  • Headless / systemd (no dashboard):      NO_TUI=true
