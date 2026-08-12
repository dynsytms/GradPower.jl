#!/bin/bash
# environment setup for polaris, provisioning needs the compute-node proxy, so run it inside a PBS job
#
# run scripts automatically run the setup if needed, but the first bootstrap costs ~5-10 min (download + precompile).
# afterwards it is a few file checks and returns immediately.

_gp_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GRADPOWER_DIR="${GRADPOWER_DIR:-$(cd "$_gp_here/../.." && pwd)}"

source "$_gp_here/polaris_env.sh"

_gp_stamp="$GP_ENV/.gp_bootstrap"   # CUDSS_VERSION/CUDSS_ROOT come from polaris_env.sh

_gp_log() { echo "[bootstrap] $*"; }

_gp_have_cudss() { [[ -f "$JULIA_CUDSS_LIBRARY_PATH/libcudss.so" ]]; }

_gp_env_ready() {
    [[ -f "$GP_ENV/Manifest.toml" && -f "$_gp_stamp" ]] || return 1
    grep -qxF "repo=$GRADPOWER_DIR" "$_gp_stamp" || return 1
    grep -qxF "cudss=$JULIA_CUDSS_LIBRARY_PATH" "$_gp_stamp" || return 1
    return 0
}

_gp_fetch_cudss() {
    local archive="libcudss-linux-x86_64-${CUDSS_VERSION}-archive"
    local tarball="${archive}.tar.xz"
    local url="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/${tarball}"
    local dest
    dest="$(dirname "$(dirname "$JULIA_CUDSS_LIBRARY_PATH")")"   # .../cudss

    _gp_log "fetching cuDSS $CUDSS_VERSION -> $dest/$archive"
    mkdir -p "$dest" || return 1
    ( cd "$dest" && curl -fL -o "$tarball" "$url" && tar xf "$tarball" && rm -f "$tarball" ) || {
        _gp_log "ERROR: cuDSS download failed. On a compute node the proxy must be set"
        _gp_log "       (http_proxy=$http_proxy). Versions:"
        _gp_log "       https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/"
        return 1
    }
    _gp_have_cudss || {
        _gp_log "ERROR: no libcudss.so under $JULIA_CUDSS_LIBRARY_PATH after extraction."
        _gp_log "       CUDSS_VERSION=$CUDSS_VERSION may not match CUDSS_ROOT=$CUDSS_ROOT."
        return 1
    }
}

_gp_build_env() {
    _gp_log "building Julia environment $GP_ENV (dev GradPower + CUDA + CUDSS)"
    "${JULIA:-julia}" --project="$GP_ENV" "$_gp_here/polaris_setup.jl" || return 1
    printf 'repo=%s\ncudss=%s\n' "$GRADPOWER_DIR" "$JULIA_CUDSS_LIBRARY_PATH" > "$_gp_stamp"
}

_gp_status() {
    _gp_log "repo   : $GRADPOWER_DIR"
    _gp_log "GP_ENV : $GP_ENV        $(_gp_env_ready && echo ready || echo 'NOT ready')"
    _gp_log "cuDSS  : $JULIA_CUDSS_LIBRARY_PATH  $(_gp_have_cudss && echo found || echo MISSING)"
}

_gp_force=0
_gp_check=0

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    for _a in "$@"; do
        case "$_a" in
            --force) _gp_force=1 ;;
            --check) _gp_check=1 ;;
            *) echo "usage: $0 [--check|--force]" >&2; exit 2 ;;
        esac
    done
fi

_gp_rc=0
if [[ "$_gp_check" == 1 ]]; then
    _gp_status
    _gp_env_ready && _gp_have_cudss || _gp_rc=1
elif [[ "${GP_BOOTSTRAP:-1}" == "0" ]]; then
    _gp_log "GP_BOOTSTRAP=0 — skipping provisioning, using the environment as-is"
    _gp_status
else
    if [[ "$_gp_force" == 1 ]] || ! _gp_have_cudss; then
        _gp_fetch_cudss || _gp_rc=1
    fi
    if [[ "$_gp_rc" == 0 ]] && { [[ "$_gp_force" == 1 ]] || ! _gp_env_ready; }; then
        _gp_build_env || _gp_rc=1
    fi
    if [[ "$_gp_rc" == 0 ]]; then
        _gp_log "environment ready (GP_ENV=$GP_ENV)"
    else
        _gp_log "BOOTSTRAP FAILED — see errors above"
        _gp_status
    fi
fi

unset _gp_here _gp_force _gp_check _a
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then exit "$_gp_rc"; else return "$_gp_rc"; fi
