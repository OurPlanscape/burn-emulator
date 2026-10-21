# sourced by the scripts and slurm jobs: exports ~/.config/burn-emulator/env (make env-vars;
# BURN_EMULATOR_ENV_FILE to override), the source of truth for BURN_EMULATOR_* and
# GOOGLE_APPLICATION_CREDENTIALS. Skipped under make, and in jobs it submitted: make already exported
# the file, with its own command-line values on top.
if [[ -z "${BURN_EMULATOR_ENV_LOADED:-}" ]]; then
    _env_file="${BURN_EMULATOR_ENV_FILE:-$HOME/.config/burn-emulator/env}"
    if [[ -f "$_env_file" ]]; then
        set -a
        # shellcheck source=/dev/null
        source "$_env_file"
        set +a
    fi
    export BURN_EMULATOR_ENV_LOADED=1
    unset _env_file
fi
