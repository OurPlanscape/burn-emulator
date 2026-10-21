import logging
import os
import signal
import sys
import tempfile

from burn_emulator_runner.job import (
    REPORT_COMPLETED,
    REPORT_FAILED,
    publish,
    run_job,
    write_report,
)
from burn_emulator_runner.utils import env_flag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("burn_emulator_runner")


def _on_sigterm(signum, frame):
    raise TimeoutError("received SIGTERM (task timeout or cancellation)")


# Cloud Run: one execution, one treatment area, env set by the api's trigger
def _run_once() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)
    run_hash = os.environ.get("BURN_EMULATOR_HASH", "<unknown>")
    report_path = os.environ.get("BURN_EMULATOR_REPORT_PATH", "")
    claim_generation = os.environ.get("BURN_EMULATOR_CLAIM_GENERATION", "")
    try:
        with tempfile.TemporaryDirectory() as local_dir:
            out = run_job(os.environ, local_dir)
            # SIGTERM is ignored from here on
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            publish(out)

        log.info("run done hash=%s output_path=%s", run_hash, out.output_path)
        write_report(report_path, claim_generation, REPORT_COMPLETED)
    except Exception as e:
        # SIGTERM is ignored while reporting the failure
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        log.exception("run failed hash=%s", run_hash)
        write_report(report_path, claim_generation, REPORT_FAILED, f"{type(e).__name__}: {e}")
        sys.exit(1)


def main() -> None:
    # slurm: one long-lived process takes queued claims until stopped
    if env_flag("BURN_EMULATOR_LOOP"):
        from burn_emulator_runner.loop import serve

        serve()
    else:
        _run_once()


if __name__ == "__main__":
    main()
