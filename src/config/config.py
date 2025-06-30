from viam.proto.app.robot import ServiceConfig

from src.config.attribute import (
    BoolAttribute,
    ChosenLabelsAttribute,
    DictAttribute,
    FloatAttribute,
    IntAttribute,
    StringAttribute,
)


class TrackingConfig:
    def __init__(self, config: "ServiceConfig"):
        self.lambda_value = FloatAttribute(
            field_name="lambda_value",
            min_value=0,
            max_value=1,
            default_value=0.0005,  # TODO: change that when we choose our embedder
        ).validate(config)
        self.max_age_track = IntAttribute(
            field_name="max_age_track",
            min_value=0,
            max_value=100000,
            default_value=1000,
        ).validate(config)
        self.embedder_threshold = FloatAttribute(
            field_name="embedder_threshold",
            min_value=0,
            max_value=1,
            default_value=0.3,
        ).validate(config)

        self.max_frequency = FloatAttribute(
            field_name="max_frequency_hz",
            default_value=10,
            min_value=0.1,
            max_value=100,
        ).validate(config)

        self._start_background_loop = BoolAttribute(
            field_name="_start_background_loop", default_value=True
        ).validate(config)

        self.crop_region = DictAttribute(
            field_name="crop_region",
            default_value=None,
            fields=[
                FloatAttribute(field_name="x1_rel", min_value=0, max_value=1),
                FloatAttribute(field_name="y1_rel", min_value=0, max_value=1),
                FloatAttribute(field_name="x2_rel", min_value=0, max_value=1),
                FloatAttribute(field_name="y2_rel", min_value=0, max_value=1),
            ],
        ).validate(config)


class DetectorConfig:
    def __init__(self, config: "ServiceConfig"):
        self.detector_name = StringAttribute(
            field_name="detector_name",
            default_value=None,
        ).validate(config)
        self.chosen_labels = ChosenLabelsAttribute(
            field_name="chosen_labels",
            default_value=None,
        ).validate(config)
        self.device = StringAttribute(
            field_name="detector_device",
            default_value="cpu",
            allowlist=["cpu", "cuda"],  # TODO: can add MPS backend here if we want
        ).validate(config)
        self._enable_debug_tools = BoolAttribute(
            field_name="_enable_debug_tools", default_value=False
        ).validate(config)

        self._path_to_debug_directory = StringAttribute(
            field_name="_path_to_debug_directory",
            default_value=None,
        ).validate(config)

        self._max_size_debug_directory = IntAttribute(
            field_name="_max_size_debug_directory", default_value=200
        ).validate(config)


class EmbedderConfig:
    def __init__(self, config: ServiceConfig):
        self.embedder_name = StringAttribute(
            field_name="embedder_model",
            default_value=None,
        ).validate(config)

        self.embedder_distance = StringAttribute(
            field_name="embedder_distance",
            default_value="cosine",
            allowlist=["cosine", "euclidean", "manhattan"],
        ).validate(config)

        # NOT DEFINITIVE ARGUMENTS
        self.input_height = IntAttribute(
            field_name="embedder_input_height",
            default_value=112,
        ).validate(config)

        self.input_width = IntAttribute(
            field_name="embedder_input_width",
            default_value=112,
        ).validate(config)

        self.input_name = StringAttribute(
            field_name="embedder_input_name",
            default_value="input",
        ).validate(config)

        self.output_name = StringAttribute(
            field_name="embedder_output_name",
            default_value="output",
        ).validate(config)

        self.device = StringAttribute(
            field_name="embedder_device",
            default_value="cuda",
            allowlist=["cpu", "cuda"],
        ).validate(config)


class TrackerConfig:
    def __init__(self, config: ServiceConfig):
        self.config = config

        self.tracker_config = TrackingConfig(config)
        self.detector_config = DetectorConfig(config)
        self.embedder_config = EmbedderConfig(config)
