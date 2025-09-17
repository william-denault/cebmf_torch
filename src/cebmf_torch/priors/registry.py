from .base import PriorBuilder
from .learned import LearnedBuilder, LearnedPriorType
from .mixture import ASHBuilder, MixturePriorType
from .point import PointBuilder, PointPriorType


class PriorRegistry:
    """
    Registry for prior builders, allowing registration and lookup by name.

    Attributes
    ----------
    registry : dict[str, PriorBuilder]
        Mapping from prior name to builder instance.
    """

    registry: dict[str, PriorBuilder] = {}

    @classmethod
    def register(cls, name: str, builder: PriorBuilder):
        """
        Register a prior builder with a given name.

        Parameters
        ----------
        name : str
            Name of the prior.
        builder : PriorBuilder
            Builder instance to register.
        """
        cls.registry[name] = builder

    @classmethod
    def get_builder(cls, name: str) -> PriorBuilder:
        """
        Retrieve a registered prior builder by name.

        Parameters
        ----------
        name : str
            Name of the prior.

        Returns
        -------
        PriorBuilder
            The registered builder instance.

        Raises
        ------
        ValueError
            If the prior is not registered.
        """
        if name not in cls.registry:
            raise ValueError(f"Prior '{name}' is not registered.")
        return cls.registry[name]

    @classmethod
    def list_priors(cls) -> list[str]:
        """
        List all registered prior names.

        Returns
        -------
        list of str
            Names of all registered priors.
        """
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
