#Expose samples under codetector.samples

#These are concrete implementations
from ..src.features.shared.data.models.code_samples import CodeSampleModel as CodeSample
from ..src.features.shared.data.models.code_detection_sample_model import CodeDetectionSampleModel as CodeDetectionSample

__all__ = ['CodeSample','CodeDetectionSample']