from .metrics import KFactorMetricsCalculator
from .plotter import KFactorPlotter

class KFactorAnalyzer:
    """
    Facade para manter a compatibilidade da interface antiga e atuar como orquestrador.
    Princípio da Responsabilidade Única (SRP): Delega o trabalho de cálculo para o KFactorMetricsCalculator 
    e o trabalho de visualização para o KFactorPlotter.
    """
    def __init__(self, model_name, output_dir="analysis_outputs"):
        self.metrics_calculator = KFactorMetricsCalculator(model_name, output_dir)
        self.plotter = KFactorPlotter(model_name, output_dir)

    def calculate_statistical_metrics(self, df):
        return self.metrics_calculator.calculate_statistical_metrics(df)

    def plot_kt_kd_relationship(self, df):
        suffixes = self.metrics_calculator.get_model_suffixes(df)
        self.plotter.plot_kt_kd_relationship(df, suffixes)

    def plot_clear_sky_day_analysis(self, df, min_samples=10):
        suffixes = self.metrics_calculator.get_model_suffixes(df)
        self.plotter.plot_clear_sky_day_analysis(df, suffixes, min_samples)

    def plot_high_variability_day_analysis(self, df, min_samples=10):
        suffixes = self.metrics_calculator.get_model_suffixes(df)
        self.plotter.plot_high_variability_day_analysis(df, suffixes, min_samples)

    def plot_overcast_day_analysis(self, df, min_samples=10):
        suffixes = self.metrics_calculator.get_model_suffixes(df)
        self.plotter.plot_overcast_day_analysis(df, suffixes, min_samples)

    def plot_transient_day_analysis(self, df, min_samples=10):
        suffixes = self.metrics_calculator.get_model_suffixes(df)
        self.plotter.plot_transient_day_analysis(df, suffixes, min_samples)

    def plot_scatter_validation(self, df):
        suffixes = self.metrics_calculator.get_model_suffixes(df)
        self.plotter.plot_scatter_validation(df, suffixes)
