from .base import PriorBuilder
from .learned import LearnedBuilder, LearnedPriorType
from .mixture import ASHBuilder, MixturePriorType
from .point import PointBuilder, PointPriorType


class PriorRegistry:
    registry: dict[str, PriorBuilder] = {}

    @classmethod
    def register(cls, name: str, builder: PriorBuilder):
        cls.registry[name] = builder

    @classmethod
    def get_builder(cls, name: str) -> PriorBuilder:
        if name not in cls.registry:
            raise ValueError(f"Prior '{name}' is not registered.")
        return cls.registry[name]

    @classmethod
    def list_priors(cls) -> list[str]:
        return list(cls.registry.keys())


PRIOR_REGISTRY = PriorRegistry()

for prior_type in MixturePriorType:
    builder = ASHBuilder(prior_type)
    PRIOR_REGISTRY.register(builder.name, builder)

for prior_type in PointPriorType:
    builder = PointBuilder(prior_type)
    PRIOR_REGISTRY.register(builder.name, builder)

for prior_type in LearnedPriorType:
    builder = LearnedBuilder(prior_type)
    PRIOR_REGISTRY.register(builder.name, builder)
