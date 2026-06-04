import os
import json
import time
from datetime import datetime


class RunLogger:
    def __init__(self, log_dir, run_name, hparams: dict):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{run_name}_{timestamp}.json")
        self.data = {"run_name": run_name, "timestamp": timestamp,
                     "hparams": hparams, "epochs": []}
        self._start = time.time()
        self._epoch_start = None
        self._save()
        print(f"Logging to {self.log_path}")

    def start_epoch(self):
        self._epoch_start = time.time()

    def log_epoch(self, epoch: int, metrics: dict):
        epoch_secs = round(time.time() - self._epoch_start, 1) if self._epoch_start else None
        total_secs = round(time.time() - self._start, 1)
        self.data["epochs"].append({
            "epoch": epoch,
            "epoch_time_s": epoch_secs,
            "total_time_s": total_secs,
            **metrics,
        })
        self.data["total_time_s"] = total_secs
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)
