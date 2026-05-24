from sklearn.pipeline import Pipeline
from .transformers import (
    DataCleaningTransformer,
    SolarPositionTransformer,
    ClearSkyTransformer,
    PhysicalFeaturesTransformer,
    QualityControlTransformer,
    FeatureEngineeringTransformer,
    ScalerTransformer
)

def build_solar_pipeline(
    latitude: float,
    longitude: float,
    altitude: float = 0,
    timezone: str = 'UTC',
    nominal_power: float = 156.0,
    start_year: int = 2018,
    cs_model: str = 'esra',
    degradation_rate: float = 0.05,
    column_mapping: dict = None,
    features_to_scale: list = None,
    target_col: str = 'power',
    kasten_corr: bool = False,
    auto_identify_thermal_params: bool = True
) -> Pipeline:
    """
    Constrói e retorna o pipeline completo de pré-processamento solar
    usando transformadores desacoplados.
    """
    
    # 1. Limpeza inicial e padronização do índice
    cleaner = DataCleaningTransformer(
        column_mapping=column_mapping, 
        timezone=timezone, 
        start_year=start_year
    )
    
    # 2. Posição Solar e Variáveis Extra-terrestres
    solar_position = SolarPositionTransformer(
        latitude=latitude, 
        longitude=longitude, 
        altitude=altitude, 
        timezone=timezone
    )
    
    # 3. Modelagem de Céu Claro e Linke Turbidity
    clear_sky = ClearSkyTransformer(
        cs_model=cs_model, 
        location=solar_position.location  # reaproveitando o Location criado
    )
    
    # 4. Variáveis Físicas (Kt, frações, variação, U0/U1 otimização)
    phys_features = PhysicalFeaturesTransformer(
        nominal_power=nominal_power,
        auto_identify=auto_identify_thermal_params,
        target_col=target_col,
        kasten_corr=kasten_corr
    )
    
    # A lista base de passos do pipeline
    steps = [
        ('data_cleaning', cleaner),
        ('solar_position', solar_position),
        ('clear_sky', clear_sky),
        ('physical_features', phys_features)
    ]
    
    # 5. Controle de Qualidade (BSRN / Castillejo-Cuberos)
    # Segundo o código original, o controle de qualidade ocorre apenas se target_col == 'sky'
    if target_col == 'sky':
        steps.append(('quality_control', QualityControlTransformer()))
        
    # 6. Engenharia de Features (Lags, Persistência, etc.)
    feat_engineering = FeatureEngineeringTransformer(
        target_col=target_col,
        nominal_power=nominal_power,
        start_year=start_year,
        degradation_rate=degradation_rate
    )
    steps.append(('feature_engineering', feat_engineering))
    
    # 7. Escalador (MinMaxScaler) final
    if features_to_scale:
        scaler = ScalerTransformer(features_to_scale=features_to_scale)
        steps.append(('scaler', scaler))
        
    # Retorna o Pipeline Montado
    return Pipeline(steps)
