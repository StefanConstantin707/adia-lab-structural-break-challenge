import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Union, List, Tuple

from src.data.data_utility import get_sliding_windows_from_sequence
from src.features.base_feature_extractor import BaseFeatureExtractor
from src.features.extract_distribution_features import DistributionCombinedFeatureExtractor
from src.data.dataLoader import TimeSeriesData
from src.models.deep_learning.CNN_autoencoder import CNNTemporalAutoencoder


class CNNBreakDetectionExtractor(BaseFeatureExtractor):
    """
    CNN-based autoencoder feature extractor for break detection using DistributionCombinedFeatureExtractor
    for comprehensive statistical analysis of reconstruction patterns.
    """

    def __init__(self,
                 sequence_length: int = 128,
                 dim_raise: int = 4,
                 bottleneck_dim: int = 32,
                 train_ratio: float = 0.7,
                 num_epochs: int = 20,
                 batch_size: int = 32,
                 lr: float = 1e-2,
                 weight_decay: float = 1e-4,
                 eta_min: float = 1e-5,
                 cache_name: str = 'cnn_autoencoder_features',
                 force_recompute: bool = False,
                 check_same: bool = False,
                 **kwargs):
        """
        Args:
            sequence_length: Length of sliding windows for CNN
            dim_raise: Dimension raise parameter for CNN autoencoder
            bottleneck_dim: Bottleneck dimension for CNN autoencoder
            train_ratio: Ratio of regime 1 to use for training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            eta_min: Minimum learning rate for cosine annealing
            cache_name: Name for caching features
            force_recompute: If True, ignore cache and recompute
            check_same: Whether to check for same time series
        """
        super().__init__(
            cache_name=cache_name,
            force_recompute=force_recompute,
            check_same=check_same,
            sequence_length=sequence_length,
            dim_raise=dim_raise,
            bottleneck_dim=bottleneck_dim,
            train_ratio=train_ratio,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            eta_min=eta_min,
            **kwargs
        )

        self.sequence_length = sequence_length
        self.dim_raise = dim_raise
        self.bottleneck_dim = bottleneck_dim
        self.train_ratio = train_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_min = eta_min

        # Initialize distribution feature extractors for different comparisons
        # Use higher minimum sample size to avoid warnings
        self.dist_extractors = {
            'errors_r1_r2': DistributionCombinedFeatureExtractor(
                cache_name='temp_errors_r1_r2', force_recompute=True, min_sample_size=10
            ),
            'errors_r1_trans': DistributionCombinedFeatureExtractor(
                cache_name='temp_errors_r1_trans', force_recompute=True, min_sample_size=10
            ),
            'errors_r2_trans': DistributionCombinedFeatureExtractor(
                cache_name='temp_errors_r2_trans', force_recompute=True, min_sample_size=10
            ),
            'corrs_r1_r2': DistributionCombinedFeatureExtractor(
                cache_name='temp_corrs_r1_r2', force_recompute=True, min_sample_size=10
            ),
            'corrs_r1_trans': DistributionCombinedFeatureExtractor(
                cache_name='temp_corrs_r1_trans', force_recompute=True, min_sample_size=10
            ),
            'corrs_r2_trans': DistributionCombinedFeatureExtractor(
                cache_name='temp_corrs_r2_trans', force_recompute=True, min_sample_size=10
            ),
        }

        # Device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names produced by this extractor."""
        feature_names = []

        # Get base feature names from distribution extractor
        base_dist_features = self.dist_extractors['errors_r1_r2'].get_feature_names()

        # Add prefixes for each comparison type
        for comparison_type in ['errors_r1_r2', 'errors_r1_trans', 'errors_r2_trans',
                                'corrs_r1_r2', 'corrs_r1_trans', 'corrs_r2_trans']:
            for feature_name in base_dist_features:
                feature_names.append(f"{comparison_type}_{feature_name}")

        # Add CNN training-specific features
        feature_names.extend([
            'cnn_final_train_loss',
            'cnn_final_test_loss',
            'cnn_convergence_rate',
            'cnn_training_stability',
            'regime1_holdout_length',
            'regime2_length',
            'transition_length'
        ])

        return feature_names

    def train_cnn_autoencoder(self, regime_data: np.ndarray) -> Dict[str, float]:
        """Train CNN autoencoder on regime data with exact parameters."""

        # Create sliding windows
        sequences = get_sliding_windows_from_sequence(regime_data, self.sequence_length)

        # Scale by standard deviation
        scale_factor = np.std(sequences) + 1e-8
        sequences_scaled = sequences / scale_factor
        self.scale_factor = scale_factor

        # Create CNN autoencoder with your parameters
        self.autoencoder = CNNTemporalAutoencoder(
            sequence_length=self.sequence_length,
            bottleneck_dim=self.bottleneck_dim,
            dim_raise=self.dim_raise
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Convert to tensors
        X_train = torch.from_numpy(sequences_scaled).float().to(self.device)
        N_train = X_train.size(0)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=self.eta_min)

        # Training loop (matching exact approach)
        self.autoencoder.train()
        train_losses = []

        for epoch in range(1, self.num_epochs + 1):
            # Shuffle GPU indices once per epoch
            perm = torch.randperm(N_train, device=self.device)
            epoch_train_loss = 0.0

            # Iterate in-GPU batches
            for i in range(0, N_train, self.batch_size):
                batch_idx = perm[i:i + self.batch_size]
                batch = X_train[batch_idx]

                optimizer.zero_grad()
                recon = self.autoencoder(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * batch.size(0)

            epoch_train_loss /= N_train
            scheduler.step()
            train_losses.append(epoch_train_loss)

        self.training_losses = train_losses

        return {
            'final_train_loss': train_losses[-1],
            'n_sequences': len(sequences),
            'convergence_rate': self._calculate_convergence_rate(train_losses)
        }

    def get_reconstruction_data(self, time_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get reconstruction errors and correlations for a time series."""
        if not hasattr(self, 'autoencoder') or self.autoencoder is None:
            raise ValueError("CNN autoencoder not trained yet")

        sequences = get_sliding_windows_from_sequence(time_series, self.sequence_length)
        sequences_scaled = sequences / self.scale_factor

        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(sequences_scaled).float().to(self.device)

            # Process in batches for memory efficiency
            reconstructed_batches = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i + self.batch_size]
                recon_batch = self.autoencoder(batch)
                reconstructed_batches.append(recon_batch.cpu().numpy())

            reconstructed = np.concatenate(reconstructed_batches, axis=0)

        # Calculate reconstruction errors (MSE per sequence)
        reconstruction_errors = np.mean((sequences_scaled - reconstructed) ** 2, axis=1)

        # Calculate correlations per sequence
        correlations = []
        for i in range(len(sequences_scaled)):
            corr = np.corrcoef(sequences_scaled[i], reconstructed[i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        correlations = np.array(correlations)

        return reconstruction_errors, correlations

    def _create_comparison_data(self, data1: np.ndarray, data2: np.ndarray,
                                series_id: int = -1) -> TimeSeriesData:
        """Create TimeSeriesData object for distribution comparison."""
        # Create a synthetic time series where period_0 = data1, period_1 = data2
        # This allows us to use DistributionCombinedFeatureExtractor

        full_series = np.concatenate([data1, data2])
        time_points = np.arange(len(full_series))
        boundary_point = len(data1)

        synthetic_data = TimeSeriesData(
            series_id=series_id,
            values=full_series,
            time_points=time_points,
            boundary_point=boundary_point,
            period_0_values=data1,
            period_1_values=data2,
            has_break=True
        )

        return synthetic_data

    def _calculate_convergence_rate(self, losses: List[float]) -> float:
        """Calculate convergence rate from training losses."""
        if len(losses) < 5:
            return 0.0

        # Fit exponential decay to losses
        x = np.arange(len(losses))
        y = np.array(losses)

        try:
            # Avoid log of negative numbers
            y_positive = np.maximum(y, 1e-8)
            log_y = np.log(y_positive)
            slope = np.polyfit(x, log_y, 1)[0]
            return -slope  # Negative slope means convergence
        except:
            return 0.0

    def _compute_features(self, data: Union[pd.DataFrame, TimeSeriesData]) -> Dict[str, float]:
        """Compute all CNN-based break detection features."""
        features = {}

        try:
            # Extract period values
            if isinstance(data, TimeSeriesData):
                regime1 = data.period_0_values
                regime2 = data.period_1_values
                full_series = data.values
                break_point = data.boundary_point
                series_id = data.series_id
            else:
                # Handle DataFrame input
                regime1 = data.loc[data['period'] == 0, 'value'].values
                regime2 = data.loc[data['period'] == 1, 'value'].values
                full_series = data['value'].values
                break_point = len(regime1)
                series_id = -1

            # Train on partial regime 1
            train_size = int(self.train_ratio * len(regime1))
            train_data = regime1[:train_size]
            regime1_holdout = regime1[train_size:]

            # Train CNN autoencoder
            training_info = self.train_cnn_autoencoder(train_data)

            # Add CNN training features
            features['cnn_final_train_loss'] = training_info['final_train_loss']
            features['cnn_convergence_rate'] = training_info['convergence_rate']
            features['cnn_training_stability'] = np.std(self.training_losses[-5:]) if len(self.training_losses) >= 5 else 0.0

            # Get reconstruction data for different segments
            errors_r1, corrs_r1 = self.get_reconstruction_data(regime1_holdout)
            errors_r2, corrs_r2 = self.get_reconstruction_data(regime2)

            # Transition region (using your calculation)
            transition_size = max(self.sequence_length // 2 + 1,
                                  min(self.sequence_length * 2, len(regime1) // 4, len(regime2) // 4))

            transition_data = full_series[break_point - transition_size:break_point + transition_size]
            errors_trans, corrs_trans = self.get_reconstruction_data(transition_data)

            # Add length features
            features['regime1_holdout_length'] = len(regime1_holdout)
            features['regime2_length'] = len(regime2)
            features['transition_length'] = len(transition_data)

            # Evaluate test loss on regime2 (unseen data)
            test_sequences = get_sliding_windows_from_sequence(regime2, self.sequence_length)
            test_sequences_scaled = test_sequences / self.scale_factor
            X_test = torch.from_numpy(test_sequences_scaled).float().to(self.device)

            self.autoencoder.eval()
            with torch.no_grad():
                test_loss = 0.0
                criterion = nn.MSELoss(reduction='mean')
                for i in range(0, len(X_test), self.batch_size):
                    batch = X_test[i:i + self.batch_size]
                    recon = self.autoencoder(batch)
                    loss = criterion(recon, batch)
                    test_loss += loss.item() * batch.size(0)
                test_loss /= len(X_test)

            features['cnn_final_test_loss'] = test_loss

            # Use DistributionCombinedFeatureExtractor for 6 different comparisons
            comparisons = [
                ('errors_r1_r2', errors_r1, errors_r2),
                ('errors_r1_trans', errors_r1, errors_trans),
                ('errors_r2_trans', errors_r2, errors_trans),
                ('corrs_r1_r2', corrs_r1, corrs_r2),
                ('corrs_r1_trans', corrs_r1, corrs_trans),
                ('corrs_r2_trans', corrs_r2, corrs_trans),
            ]

            for comparison_name, data1, data2 in comparisons:
                # Create synthetic TimeSeriesData for comparison
                comparison_data = self._create_comparison_data(data1, data2, series_id)

                # Extract distribution features
                dist_features = self.dist_extractors[comparison_name]._compute_features(comparison_data)

                # Add prefix to feature names
                for feature_name, feature_value in dist_features.items():
                    features[f"{comparison_name}_{feature_name}"] = feature_value

        except Exception as e:
            print(f"Feature extraction failed: {e}")
            # Return default features on failure
            return self._get_default_features()

        return features


    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when extraction fails."""
        default_features = {}

        # Add default values for all expected features
        feature_names = self.get_feature_names()

        for name in feature_names:
            if 'pvalue' in name:
                default_features[name] = 1.0
            elif 'ratio' in name:
                default_features[name] = 1.0
            elif 'length' in name:
                default_features[name] = 0
            else:
                default_features[name] = 0.0

        return default_features





