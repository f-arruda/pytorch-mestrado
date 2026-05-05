from sklearn.pipeline import Pipeline
from typing import Optional, Dict, List
from .transformers import (
    DataSanitizer,
    SolarPositionCalculator,
    ClearSkyEstimator,
    PhysicalFeatureGenerator,
    KastenCorrector,
    QualityControlFilter,
    TemporalFeatureGenerator,
    FeatureScaler
)

def build_preprocessing_pipeline(
    latitude: float, 
    longitude: float, 
    altitude: float = 0, 
    timezone: str = 'UTC', 
    nominal_power: float = 156.0, 
    start_year: int = 2018,
    cs_model: str = 'esra',
    degradation_rate: float = 0.05,
    column_mapping: Optional[Dict[str, str]] = None,
    features_to_scale: Optional[List[str]] = None,
    target_col: str = 'power',
    kasten_corr: bool = False,
    auto_identify_thermal_params: bool = True
) -> Pipeline:
    """
    Factory function to build the Scikit-Learn preprocessing pipeline.
    """
    steps = [
        ('sanitizer', DataSanitizer(
            timezone=timezone, 
            start_year=start_year, 
            column_mapping=column_mapping
        )),
        ('solar_position', SolarPositionCalculator(
            latitude=latitude, 
            longitude=longitude, 
            altitude=altitude, 
            timezone=timezone
        )),
        ('clear_sky', ClearSkyEstimator(
            latitude=latitude, 
            longitude=longitude, 
            altitude=altitude, 
            timezone=timezone, 
            cs_model=cs_model
        )),
        ('physical_features', PhysicalFeatureGenerator(
            nominal_power=nominal_power, 
            auto_identify_thermal_params=auto_identify_thermal_params, 
            target_col=target_col
        )),
        ('kasten_correction', KastenCorrector(
            kasten_corr=kasten_corr, 
            target_col=target_col
        )),
        ('quality_control', QualityControlFilter(
            target_col=target_col
        )),
        ('temporal_features', TemporalFeatureGenerator(
            target_col=target_col
        )),
        ('scaler', FeatureScaler(
            target_col=target_col, 
            nominal_power=nominal_power, 
            start_year=start_year, 
            degradation_rate=degradation_rate, 
            features_to_scale=features_to_scale
        ))
    ]
    
    return Pipeline(steps)
