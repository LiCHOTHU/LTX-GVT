from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from ltx_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "video_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "audio_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(
        self, data_root: str | list[str], data_sources: dict[str, str] | list[str] | None = None
    ) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.
        Args:
            data_root: Root directory containing preprocessed data, OR a list of such
              roots. With a list, samples from every root are concatenated (each root
              is scanned independently and must hold all configured sources for its
              own files). This lets training read an immutable base root plus a small
              per-round additions root without materializing a symlink mirror.
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "reference_latents"])
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
        """
        super().__init__()

        roots = list(data_root) if isinstance(data_root, (list, tuple)) else [data_root]
        self.data_roots = [self._setup_data_root(r) for r in roots]
        self.data_sources = self._normalize_data_sources(data_sources)
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root).expanduser().resolve()

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _source_paths_for_root(self, root: Path) -> dict[str, Path] | None:
        """Map each data-source name to its dir under `root`.

        Returns None if `root` is missing any required source dir (e.g. an additions
        root not yet populated) — such a root is skipped rather than aborting setup,
        so an empty/partial root is tolerated as long as some root has the data.
        """
        source_paths: dict[str, Path] = {}
        for dir_name in self.data_sources:
            source_path = root / dir_name
            if not source_path.exists():
                logger.info(f"Skipping data root {root}: missing source '{dir_name}'")
                return None
            source_paths[dir_name] = source_path
        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover valid samples across ALL data roots, returning ABSOLUTE paths.

        Each root is scanned independently with the same fast two-pass cross-source
        matching, then results are concatenated (root order preserved so every output
        key stays in lockstep). Storing absolute paths is what lets samples come from
        different roots (e.g. an immutable base root + a small per-round additions
        root) without a symlink mirror.
        """
        if not self.data_sources:
            raise ValueError("No data sources configured")

        merged: dict[str, list[Path]] = {output_key: [] for output_key in self.data_sources.values()}
        for root in self.data_roots:
            source_paths = self._source_paths_for_root(root)
            if source_paths is None:
                continue
            per_root = self._discover_in_root(source_paths)
            for output_key, files in per_root.items():
                merged[output_key].extend(files)
        return merged

    def _discover_in_root(self, source_paths: dict[str, Path]) -> dict[str, list[Path]]:
        """Fast two-pass cross-source sample discovery within one root -> ABSOLUTE paths.

        Globs all sources in parallel to build full-path sets, then matches the
        primary source against the others by set membership (no O(N*sources) stats).
        """
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = source_paths[data_key]

        def _glob_source(dir_name: str) -> tuple[list[Path], set[str]]:
            source_path = source_paths[dir_name]
            paths = list(source_path.glob("**/*.pt"))
            return paths, {str(p) for p in paths}

        with ThreadPoolExecutor(max_workers=len(self.data_sources)) as executor:
            glob_results = dict(
                zip(
                    self.data_sources.keys(),
                    executor.map(_glob_source, self.data_sources.keys()),
                    strict=True,
                )
            )

        # Primary source files (may be empty for a not-yet-populated root -> 0 samples).
        data_files, _ = glob_results[data_key]
        data_files.sort()

        for dir_name, (paths, _) in glob_results.items():
            logger.debug(f"Source {dir_name} @ {data_path.parent}: {len(paths)} files")

        other_path_sets = {
            dir_name: path_set for dir_name, (_, path_set) in glob_results.items() if dir_name != data_key
        }

        sample_files: dict[str, list[Path]] = {output_key: [] for output_key in self.data_sources.values()}
        valid_count = 0
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)
            all_exist = True
            for dir_name, path_set in other_path_sets.items():
                expected = self._get_expected_file_path(source_paths, dir_name, data_file, rel_path)
                if str(expected) not in path_set:
                    logger.debug(f"Skipping {data_file.name}: no matching {dir_name} file at {expected}")
                    all_exist = False
                    break
            if all_exist:
                self._fill_sample_data_files(source_paths, data_file, rel_path, sample_files)
                valid_count += 1

        skipped = len(data_files) - valid_count
        if skipped > 0:
            logger.info(
                f"Fast index @ {data_path.parent}: {valid_count} valid from {len(data_files)} ({skipped} skipped)"
            )
        return sample_files

    @staticmethod
    def _get_expected_file_path(
        source_paths: dict[str, Path], dir_name: str, data_file: Path, rel_path: Path
    ) -> Path:
        """Absolute path of the file in `dir_name` that corresponds to `data_file`."""
        source_path = source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(
        self,
        source_paths: dict[str, Path],
        data_file: Path,
        rel_path: Path,
        sample_files: dict[str, list[Path]],
    ) -> None:
        """Append one valid sample (ABSOLUTE paths) across all output keys, in lockstep."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(source_paths, dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path)

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if not sample_counts or all(count == 0 for count in sample_counts.values()):
            raise ValueError(
                f"No valid samples found in {self.data_roots} - all configured data sources "
                f"({list(self.data_sources)}) must have matching files (per-source counts: {sample_counts})"
            )

        # Verify all output keys have the same number of samples
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            file_path = self.sample_files[output_key][index]  # absolute (possibly cross-root)

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)

                # Normalize video latent format if this is a latent source
                if "latent" in dir_name.lower():
                    data = self._normalize_video_latents(data)

                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data
