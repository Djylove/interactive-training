#!/usr/bin/env bash
# Create and activate a persistent Aim environment.
#
# Set AIM_SRC to an editable checkout of the companion Aim fork to obtain the
# Interactive Training `/live` workspace. If AIM_SRC is absent, stock Aim is
# installed; metric logging works, but the custom live-control panels do not.

_IT2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_IT2_ROOT}/init.sh"

export AIM_SRC="${AIM_SRC:-${_IT2_ROOT}/../aim}"
export AIM_ENV="${AIM_ENV:-${_IT2_ROOT}/.venv-aim}"

if [[ ! -f "$AIM_ENV/bin/activate" ]]; then
    echo "[init_aim] creating aim venv at $AIM_ENV (inherits container site-packages)"
    python -m venv --system-site-packages "$AIM_ENV"
fi
source "$AIM_ENV/bin/activate"

if ! python -c "import aim" > /dev/null 2>&1; then
    if [[ -f "$AIM_SRC/pyproject.toml" || -f "$AIM_SRC/setup.py" ]]; then
        echo "[init_aim] installing aim fork (editable) from $AIM_SRC"
        pip install -e "$AIM_SRC"
    else
        echo "[init_aim] no fork at $AIM_SRC; installing stock aim>=3.17"
        pip install "aim>=3.17"
    fi
fi

unset _IT2_ROOT
