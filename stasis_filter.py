import numpy as np
from typing import List, Union, Dict, Optional
from abc import ABC, abstractmethod

class PredictionModel(ABC):
    """Abstract base class for prediction models"""
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions"""
        pass

class StasisFilter:
    """
    A class to filter and stabilize probability predictions across multiple models.
    
    Attributes:
        threshold (float): Maximum allowed deviation in probabilities
        window_size (int): Number of predictions to consider for stasis
        models (List[PredictionModel]): List of prediction models
        history (Dict): History of predictions for each model
    """
    
    def __init__(self, 
                 threshold: float = 0.1,
                 window_size: int = 3,
                 models: Optional[List[PredictionModel]] = None):
        """
        Initialize the StasisFilter.
        
        Args:
            threshold (float): Maximum allowed probability deviation (default: 0.1)
            window_size (int): Number of predictions to track (default: 3)
            models (List[PredictionModel], optional): List of prediction models
        """
        self.threshold = threshold
        self.window_size = window_size
        self.models = models if models is not None else []
        self.history = {i: [] for i in range(len(self.models))}
        
    def add_model(self, model: PredictionModel) -> None:
        """Add a new prediction model to the filter"""
        self.models.append(model)
        self.history[len(self.models) - 1] = []
        
    def _check_stasis(self, model_idx: int, new_probs: np.ndarray) -> bool:
        """Check if probabilities are in stasis"""
        if len(self.history[model_idx]) < self.window_size:
            return True
            
        recent_history = np.array(self.history[model_idx][-self.window_size:])
        mean_probs = np.mean(recent_history, axis=0)
        deviation = np.max(np.abs(new_probs - mean_probs))
        
        return deviation <= self.threshold
        
    def _stabilize_probs(self, probs: np.ndarray, model_idx: int) -> np.ndarray:
        """Stabilize probabilities based on history"""
        if len(self.history[model_idx]) == 0:
            return probs
        return np.mean(self.history[model_idx], axis=0)
        
    def predict(self, X: Union[np.ndarray, List]) -> Dict[int, np.ndarray]:
        """
        Make predictions with stasis filtering.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Dict[int, np.ndarray]: Filtered predictions for each model
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        results = {}
        for idx, model in enumerate(self.models):
            probs = model.predict_proba(X)
            
            # Check stasis and update history
            if self._check_stasis(idx, probs):
                self.history[idx].append(probs)
            else:
                probs = self._stabilize_probs(probs, idx)
                self.history[idx].append(probs)
                
            # Maintain window size
            if len(self.history[idx]) > self.window_size:
                self.history[idx] = self.history[idx][-self.window_size:]
                
            results[idx] = probs
            
        return results
    
    def reset_history(self) -> None:
        """Reset prediction history"""
        self.history = {i: [] for i in range(len(self.models))}