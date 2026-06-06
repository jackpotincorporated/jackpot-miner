#!/usr/bin/env bash
# Double-click / .desktop launcher: runs jackpotminer with the mine.toml next to
# it and keeps the terminal window open afterwards so output/errors are readable.
cd "$(dirname "$(readlink -f "$0")")" || exit 1

bin="./target/release/jackpotminer"
if [ ! -x "$bin" ]; then
    echo "jackpotminer is not built yet. Run:  cargo build --release"
elif [ ! -f mine.toml ]; then
    echo "No mine.toml found. Copy mine.example.toml to mine.toml and set your"
    echo "pool URL + wallet, then double-click again."
else
    "$bin" --config mine.toml "$@"
    status=$?
    echo
    echo "jackpotminer exited (code $status)."
fi

echo
read -n1 -rsp "Press any key to close this window..."
echo
