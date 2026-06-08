from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class SignalConfig:
    sampling_hz: float = 125.0
    hrv_window_s: float = 300.0
    hrv_stride_s: float = 60.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    ibi_min_ms: float = 300.0
    ibi_max_ms: float = 2000.0
    resample_hz: float = 4.0
    temp_median_window: int = 5
    vlf_band: list[float] = field(default_factory=lambda: [0.0033, 0.04])
    lf_band: list[float] = field(default_factory=lambda: [0.04, 0.15])
    hf_band: list[float] = field(default_factory=lambda: [0.15, 0.40])


@dataclass
class DataConfig:
    name: str = "mimic_waveform"
    root: str = "data/mimic4wdb"
    synthetic: bool = True
    window_hours: float = 4.0
    time_steps: int = 236
    hrv_features: int = 12
    temp_features: int = 7
    num_features: int = 19
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    positive_rate: float = 0.219
    num_subjects: int = 2847
    batch_size: int = 64
    num_workers: int = 0
    oversample_minority: bool = True
    modalities: list[str] = field(default_factory=lambda: ["hrv", "temp"])
    signal: SignalConfig = field(default_factory=SignalConfig)


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    tcn_layers: int = 4
    tcn_kernel: int = 3
    attn_heads: int = 8
    attn_dim: int = 32
    spatial_attn_dim: int = 64
    feature_embed_dim: int = 256
    mlp_hidden: int = 512
    dropout: float = 0.3
    num_features: int = 19
    use_dilation: bool = True
    use_temporal_attention: bool = True
    use_spatial_attention: bool = True


@dataclass
class CRGAConfig:
    enabled: bool = True
    lambda_kl: float = 0.1
    lambda_smooth: float = 0.05
    warmup_epochs: int = 20
    degree_smoothing: float = 1.0


@dataclass
class ObjectiveConfig:
    use_focal: bool = True
    focal_gamma: float = 2.0
    l2_reg: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class TrainConfig:
    epochs: int = 100
    grad_accum: int = 4
    max_grad_norm: float = 0.0
    amp: bool = False
    early_stop_patience: int = 15
    early_stop_metric: str = "auroc"
    cosine_eta_min: float = 1e-6
    seed: int = 0
    threshold: float = 0.5
    ckpt_dir: str = "checkpoints"
    log_every: int = 10
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class EdgeConfig:
    distill_temperature: float = 4.0
    student_hidden: int = 64
    student_layers: int = 3
    student_heads: int = 2
    student_mlp: int = 128
    distill_alpha: float = 0.5
    distill_epochs: int = 30
    quant_bits: int = 8
    quant_symmetric: bool = True


@dataclass
class ExperimentConfig:
    name: str = "main"
    seeds: int = 5
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    crga: CRGAConfig = field(default_factory=CRGAConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)


def edge_model_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=64,
        tcn_layers=3,
        attn_heads=2,
        attn_dim=32,
        spatial_attn_dim=32,
        feature_embed_dim=64,
        mlp_hidden=128,
        dropout=0.2,
    )


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="thrombxai_config", node=ExperimentConfig)
    cs.store(group="model", name="full", node=ModelConfig())
    cs.store(group="model", name="edge", node=edge_model_config())
