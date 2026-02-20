import logging
import os

def setup_logger(run_dir: str, name: str = "election_sim", run_log_name: str = "run.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(run_dir, run_log_name), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
