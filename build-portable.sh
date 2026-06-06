#!/bin/sh
# Build a portable jackpotminer binary inside an old-glibc (Debian bullseye,
# glibc 2.31) container, so it runs on essentially any recent Linux regardless of
# the build host's glibc. CUDA is dlopen'd at runtime, so no NVIDIA toolkit is
# needed to build; the binary drives both AMD (OpenCL) and NVIDIA (CUDA) cards.
#
# Output: dist/jackpotminer
#
# Works with Docker (BuildKit) or Podman. Override the engine with ENGINE=podman.
set -eu

ENGINE="${ENGINE:-docker}"
OUT="${OUT:-dist}"

mkdir -p "$OUT"

case "$ENGINE" in
  podman)
    # Podman builds the final `scratch` stage; extract the binary from it.
    podman build -f Dockerfile.portable -t jackpotminer-portable .
    cid=$(podman create jackpotminer-portable)
    podman cp "$cid:/jackpotminer" "$OUT/jackpotminer"
    podman rm "$cid" >/dev/null
    ;;
  *)
    DOCKER_BUILDKIT=1 "$ENGINE" build -f Dockerfile.portable \
      --output "type=local,dest=$OUT" .
    ;;
esac

chmod +x "$OUT/jackpotminer"
echo
echo "Built $OUT/jackpotminer"
command -v file >/dev/null 2>&1 && file "$OUT/jackpotminer" || true
echo "Minimum glibc / dynamic deps:"
{ objdump -T "$OUT/jackpotminer" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1; } || true
