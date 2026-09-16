"""The per-block FNO surrogate: embedding, model, data, training, engine, manifests."""
from .embedding import Embedding, TorchEmbedding, block_shapes
from .fno import BlockFNO, FNOConfig
from .dataset import DataConfig, random_mixed_populations, sample_frequencies
from .train import TrainConfig, train, load_model, FingerprintMismatch
from .fno_engine import FnoEngine
from .fno_env import FnoEnv
from .manifest import Manifest, ManifestEntry, build_manifest, load_manifest, manifest_path

__all__ = ["Embedding", "TorchEmbedding", "block_shapes", "BlockFNO", "FNOConfig", "DataConfig",
           "random_mixed_populations", "sample_frequencies", "TrainConfig", "train", "load_model",
           "FingerprintMismatch", "FnoEngine", "FnoEnv", "Manifest", "ManifestEntry", "build_manifest",
           "load_manifest", "manifest_path"]
