"""Callback to `gcloud rsync` output dir to GCS at the end of each epoch"""

import os
import subprocess

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class GCloudRsyncCallback(TrainerCallback):
    def __init__(self, disk_output_dir: str, gcs_output_dir: str, gcp_project: str = None):

        # TODO: support loading model checkpoints?? -> not for now

        if not self.is_gcloud_installed():
            raise ImportError("`gcloud` commandline util is not installed. Please install it with `pip install gcsfs`.")

        # setup
        self.gcp_project = gcp_project if gcp_project is not None else os.environ.get("GCP_PROJECT")
        # TODO: token necessary?

        self.bucket, self.prefix = self.parse_gcs_path(gcs_output_dir)
        self.gcs_output_dir = gcs_output_dir
        self.disk_output_dir = disk_output_dir
        print(f"GCloudRsyncCallback:: GCP project: {self.gcp_project}, bucket: {self.bucket}, prefix: {self.prefix}")

    @staticmethod
    def parse_gcs_path(gcs_path: str):
        """gs://<bucket>/<prefix>"""
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}\nGCS path should be in the form of gs://<bucket>/<prefix>")
        gcs_path = gcs_path[5:]
        if "/" not in gcs_path:
            raise ValueError(f"Invalid GCS path: {gcs_path}\nGCS path should be in the form of gs://<bucket>/<prefix>")
        bucket, prefix = gcs_path.split("/", 1)
        print(f"GCloudRsyncCallback:: Parsed GCS path: gcs://{gcs_path} -> bucket: {bucket}, prefix: {prefix}")
        return bucket, prefix

    @staticmethod
    def is_gcloud_installed():
        try:
            subprocess.run(["gcloud", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            return False

    def rsync(self, source: str = None, dest: str = None, recursive: bool = True):
        if source is None:
            source = self.disk_output_dir
        if dest is None:
            dest = f"gs://{self.bucket}/{self.prefix}"

        cmds = ["gcloud", "alpha", "storage", "rsync", source, dest, "--project", self.gcp_project]
        if recursive:
            cmds.append("--recursive")
        print(f"GCloudRsyncCallback:: Rsyncing via `{' '.join(cmds)}`...")
        subprocess.run(cmds)

    # TODO: rsync on init in opposite direction? in case of resuming training
    # def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if state.is_world_process_zero:
    #         print("GCloudRsyncCallback:: Rsyncing at the beginning of training...")
    #         self.rsync()

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            print("GCloudRsyncCallback:: Rsyncing after saving checkpoint...")
            self.rsync()

