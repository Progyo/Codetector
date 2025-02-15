#Expose repository under codetector.models
from ..src.features.shared.domain.entities.models.base_model import BaseModel
from ..src.features.shared.domain.entities.models.generator_model import GeneratorMixin
from ..src.features.shared.domain.entities.models.detector_model import DetectorMixin

__all__ = ['BaseModel','GeneratorMixin','DetectorMixin']