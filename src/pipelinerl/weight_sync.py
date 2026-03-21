"""Weight synchronization between the Trainer and Actor processes.

This module implements the core "in-flight weight update" mechanism from
the PipelineRL paper. The key idea:

  1. Trainer saves model weights in HuggingFace format after each optimizer step.
  2. Trainer writes a version file to signal the Actor.
  3. Actor tells vLLM to reload weights from the saved directory.
     In-progress sequences continue with the new weights, creating
     "mixed-policy" sequences (early tokens from old policy, later
     tokens from new policy).

Communication is via filesystem (atomic file writes), which works
reliably across separate OS processes — including vLLM's own
EngineCore subprocess.
"""

import logging
import shutil
from pathlib import Path

import torch

logger = logging.getLogger("pipelinerl")


class WeightSynchronizer:
    """Manages weight transfer between Trainer (writer) and Actor (reader).

    The Trainer saves weights in HuggingFace save_pretrained format so
    that vLLM can reload them via its reload_weights API. A version file
    tracks the current optimizer step.
    """

    def __init__(self, sync_dir: str):
        self.sync_dir = Path(sync_dir)
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        self._last_seen_version = -1

    @property
    def hf_weight_dir(self) -> Path:
        """Directory where weights are saved in HuggingFace format."""
        return self.sync_dir / "hf_model"

    @property
    def version_path(self) -> Path:
        return self.sync_dir / "weight_version.txt"

    # -------------------------------------------------------------------------
    # Trainer side: publish new weights
    # -------------------------------------------------------------------------

    def publish_weights(self, model: torch.nn.Module, step: int) -> None:
        """Save model weights in HF format and signal the Actor.

        Called by the Trainer after each optimizer step. This is the
        'request_actor_weight_update(π)' from Algorithm 2, line 18.

        Saves in HuggingFace save_pretrained format so vLLM can
        reload via its standard weight loading pipeline.
        """
        # Save in HF format (safetensors) — this is what vLLM expects
        save_dir = self.hf_weight_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)

        # Write version file (atomic via rename)
        tmp_ver = self.sync_dir / "version_tmp.txt"
        tmp_ver.write_text(str(step))
        shutil.move(str(tmp_ver), str(self.version_path))

        logger.debug(f"Trainer published weights at step {step}")

    # -------------------------------------------------------------------------
    # Actor side: detect and acknowledge new weights
    # -------------------------------------------------------------------------

    def check_for_update(self) -> bool:
        """Check if the Trainer has published new weights (non-blocking)."""
        if not self.version_path.exists():
            return False
        try:
            current_version = int(self.version_path.read_text().strip())
            return current_version > self._last_seen_version
        except (ValueError, OSError):
            return False

    def mark_loaded(self) -> int:
        """Mark that the Actor has loaded the latest weights.

        Returns the version number of the loaded weights.
        """
        version = int(self.version_path.read_text().strip())
        self._last_seen_version = version
        return version
