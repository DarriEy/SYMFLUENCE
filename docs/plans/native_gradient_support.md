# Native Gradient Support for Gradient-Based Optimization

## Overview

Add optional native gradient support (e.g., JAX autodiff) to Adam and L-BFGS optimizers while keeping finite differences (FD) as the default. This enables ~15x speedup for JAX-based models like HBV.

## Design Principles

1. **Backward compatible**: All existing models continue working with FD
2. **Opt-in**: Models declare gradient capability; optimizer detects and uses it
3. **Configurable**: Users can force FD even when native gradients available (for comparison)
4. **Minimal changes**: Extend existing interfaces rather than replacing them

---

## Architecture Changes

### 1. Worker Interface Extension

**File**: `src/symfluence/optimization/workers/base_worker.py`

Add optional gradient methods to `BaseWorker`:

```python
class BaseWorker(ABC):
    # ... existing methods ...

    def supports_native_gradients(self) -> bool:
        """
        Check if worker supports native gradient computation.

        Returns:
            True if compute_gradient() and evaluate_with_gradient() are available.
            Default: False (use finite differences)
        """
        return False

    def compute_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Optional[Dict[str, float]]:
        """
        Compute gradient of loss w.r.t. parameters using native method (e.g., autodiff).

        Args:
            params: Current parameter values
            metric: Objective metric ('kge', 'nse', etc.)

        Returns:
            Dictionary mapping parameter names to gradient values, or None if not supported.
        """
        return None  # Default: not supported

    def evaluate_with_gradient(
        self,
        params: Dict[str, float],
        metric: str = 'kge'
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Evaluate loss and compute gradient in single pass (efficient for autodiff).

        Args:
            params: Parameter values
            metric: Objective metric

        Returns:
            Tuple of (loss_value, gradient_dict). gradient_dict is None if not supported.
        """
        # Default: evaluate only, no gradient
        # Subclasses override for native gradient support
        raise NotImplementedError("Subclass must implement for native gradients")
```

### 2. HBV Worker (Already Implemented)

**File**: `src/symfluence/models/hbv/calibration/worker.py`

HBV worker already has these methods. Just add the capability flag:

```python
@OptimizerRegistry.register_worker('HBV')
class HBVWorker(BaseWorker):

    def supports_native_gradients(self) -> bool:
        """HBV supports JAX autodiff when JAX is available."""
        return HAS_JAX

    # compute_gradient() - already implemented (lines 438-497)
    # evaluate_with_gradient() - already implemented (lines 499-551)
```

### 3. Algorithm Interface Extension

**File**: `src/symfluence/optimization/optimizers/algorithms/base_algorithm.py`

Extend the optimize signature to accept an optional gradient callback:

```python
from typing import Protocol, Optional, Tuple

class GradientCallback(Protocol):
    """Protocol for native gradient computation."""

    def __call__(
        self,
        x_normalized: np.ndarray,
        param_names: List[str]
    ) -> Tuple[float, np.ndarray]:
        """
        Compute loss and gradient.

        Args:
            x_normalized: Parameter values in [0,1] normalized space
            param_names: List of parameter names (for denormalization context)

        Returns:
            Tuple of (loss_value, gradient_array)
        """
        ...


class OptimizationAlgorithm(ABC):

    @abstractmethod
    def optimize(
        self,
        n_params: int,
        evaluate_solution: Callable[[np.ndarray, int], float],
        evaluate_population: Callable[[np.ndarray, int], np.ndarray],
        denormalize_params: Callable[[np.ndarray], Dict],
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        # NEW: Optional native gradient callback
        compute_gradient: Optional[Callable[[np.ndarray], Tuple[float, np.ndarray]]] = None,
        gradient_mode: str = 'auto',  # 'auto', 'native', 'finite_difference'
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization algorithm.

        Args:
            ...existing args...
            compute_gradient: Optional callback for native gradient computation.
                             Signature: (x_normalized) -> (loss, gradient_array)
                             If provided and gradient_mode != 'finite_difference',
                             uses this instead of FD.
            gradient_mode: How to compute gradients:
                          - 'auto': Use native if available, else FD
                          - 'native': Require native gradients (error if unavailable)
                          - 'finite_difference': Force FD even if native available
        """
        pass
```

### 4. Adam Algorithm with Native Gradient Support

**File**: `src/symfluence/optimization/optimizers/algorithms/adam.py`

```python
class AdamAlgorithm(OptimizationAlgorithm):

    def optimize(
        self,
        n_params: int,
        evaluate_solution: Callable[[np.ndarray, int], float],
        evaluate_population: Callable[[np.ndarray, int], np.ndarray],
        denormalize_params: Callable[[np.ndarray], Dict],
        record_iteration: Callable,
        update_best: Callable,
        log_progress: Callable,
        compute_gradient: Optional[Callable[[np.ndarray], Tuple[float, np.ndarray]]] = None,
        gradient_mode: str = 'auto',
        **kwargs
    ) -> Dict[str, Any]:

        # Determine gradient method
        use_native = self._should_use_native_gradients(compute_gradient, gradient_mode)

        if use_native:
            self.logger.info("Using native gradients (autodiff)")
            gradient_func = self._make_native_gradient_func(compute_gradient)
        else:
            self.logger.info("Using finite-difference gradients")
            gradient_func = self._make_fd_gradient_func(evaluate_solution, gradient_epsilon)

        # ... existing hyperparameter setup ...

        for step in range(steps):
            # Use unified gradient function
            fitness, gradient = gradient_func(x)

            # ... rest of Adam update unchanged ...

    def _should_use_native_gradients(
        self,
        compute_gradient: Optional[Callable],
        gradient_mode: str
    ) -> bool:
        """Determine whether to use native gradients."""
        if gradient_mode == 'finite_difference':
            return False
        if gradient_mode == 'native':
            if compute_gradient is None:
                raise ValueError("gradient_mode='native' but no gradient callback provided")
            return True
        # 'auto' mode
        return compute_gradient is not None

    def _make_native_gradient_func(
        self,
        compute_gradient: Callable
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """Wrap native gradient callback."""
        def gradient_func(x: np.ndarray) -> Tuple[float, np.ndarray]:
            loss, grad = compute_gradient(x)
            # Note: native returns loss (minimization), we need fitness (maximization)
            return -loss, -grad
        return gradient_func

    def _make_fd_gradient_func(
        self,
        evaluate_solution: Callable,
        epsilon: float
    ) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
        """Create finite-difference gradient function."""
        def gradient_func(x: np.ndarray) -> Tuple[float, np.ndarray]:
            return self._compute_gradients(x, evaluate_solution, epsilon)
        return gradient_func

    # Keep existing _compute_gradients for FD fallback
    def _compute_gradients(self, x, evaluate_func, epsilon):
        # ... existing FD implementation unchanged ...
```

### 5. L-BFGS Algorithm (Same Pattern)

**File**: `src/symfluence/optimization/optimizers/algorithms/lbfgs.py`

Apply the same pattern as Adam - add `compute_gradient` and `gradient_mode` parameters,
use unified gradient function internally.

### 6. BaseModelOptimizer Integration

**File**: `src/symfluence/optimization/optimizers/base_model_optimizer.py`

Wire native gradients from worker to algorithm:

```python
class BaseModelOptimizer:

    def _create_gradient_callback(self) -> Optional[Callable]:
        """
        Create gradient callback if worker supports native gradients.

        Returns:
            Callback function or None if not supported.
        """
        if not hasattr(self.worker, 'supports_native_gradients'):
            return None

        if not self.worker.supports_native_gradients():
            return None

        # Get metric from config
        metric = self._get_config_value(
            lambda: self.config.optimization.calibration_metric,
            default='kge',
            dict_key='CALIBRATION_METRIC'
        )

        def gradient_callback(x_normalized: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Compute loss and gradient from normalized parameters.

            Args:
                x_normalized: Parameters in [0,1] space

            Returns:
                Tuple of (loss, gradient_array)
            """
            # Denormalize to physical parameters
            params_dict = self.param_manager.denormalize_parameters(x_normalized)

            # Call worker's evaluate_with_gradient
            loss, grad_dict = self.worker.evaluate_with_gradient(params_dict, metric)

            if grad_dict is None:
                raise RuntimeError("Worker returned None gradient")

            # Convert gradient dict to array (same order as denormalize)
            param_names = self.param_manager.get_parameter_names()
            grad_array = np.array([grad_dict[name] for name in param_names])

            # Transform gradient from physical to normalized space
            # d(loss)/d(x_norm) = d(loss)/d(x_phys) * d(x_phys)/d(x_norm)
            # where d(x_phys)/d(x_norm) = (upper - lower) for each param
            bounds = self.param_manager.get_parameter_bounds()
            scale = np.array([
                bounds[name]['max'] - bounds[name]['min']
                for name in param_names
            ])
            grad_normalized = grad_array * scale

            return loss, grad_normalized

        return gradient_callback

    def run_optimization(self, algorithm_name: str, **kwargs) -> Path:
        """Run optimization with specified algorithm."""

        # ... existing setup ...

        # Create gradient callback if available
        gradient_callback = self._create_gradient_callback()

        # Get gradient mode from config
        gradient_mode = self._get_config_value(
            lambda: self.config.optimization.gradient_mode,
            default='auto',
            dict_key='GRADIENT_MODE'
        )

        # Log gradient method
        if gradient_callback is not None and gradient_mode != 'finite_difference':
            self.logger.info(f"Native gradient support available for {self._get_model_name()}")

        # Run algorithm with gradient callback
        result = algorithm.optimize(
            n_params=n_params,
            evaluate_solution=self._evaluate_solution,
            evaluate_population=self._evaluate_population,
            denormalize_params=self.param_manager.denormalize_parameters,
            record_iteration=self._record_iteration,
            update_best=self._update_best,
            log_progress=self._log_progress,
            compute_gradient=gradient_callback,  # NEW
            gradient_mode=gradient_mode,          # NEW
            **kwargs
        )

        # ... existing result handling ...
```

### 7. Configuration Options

**File**: `src/symfluence/core/config/models/model_configs.py` (or similar)

Add new config options:

```yaml
optimization:
  # Gradient computation method for Adam/L-BFGS
  # Options: 'auto', 'native', 'finite_difference'
  # - auto: Use native gradients if model supports them, else FD
  # - native: Require native gradients (error if not available)
  # - finite_difference: Always use FD (useful for comparison)
  gradient_mode: auto

  # Finite difference epsilon (only used when gradient_mode != 'native')
  gradient_epsilon: 1.0e-4

  # Gradient clipping (applies to both methods)
  gradient_clip_value: 1.0
```

---

## Implementation Order

### Phase 1: Core Infrastructure ✅ COMPLETED
1. ✅ Add `supports_native_gradients()` to `BaseWorker`
2. ✅ Add `compute_gradient` parameter to algorithm base class
3. ✅ Update Adam algorithm to use unified gradient interface
4. ✅ Update L-BFGS algorithm similarly

### Phase 2: HBV Integration ✅ COMPLETED
5. ✅ Add `supports_native_gradients()` to `HBVWorker` (return `HAS_JAX`)
6. ✅ Wire gradient callback in `BaseModelOptimizer`
7. ✅ Add `GRADIENT_MODE` config option

### Phase 3: Testing & Validation ✅ COMPLETED
8. ✅ Add unit tests for gradient callback
9. ✅ Add integration test comparing FD vs native gradients
10. ✅ Add benchmark script for HBV calibration
10. Benchmark: measure speedup on HBV calibration

### Phase 4: Documentation
11. Update docstrings
12. Add example showing gradient mode configuration

---

## Usage Examples

### Default (Auto Mode)
```python
# HBV will automatically use JAX autodiff if available
optimizer = HBVModelOptimizer(config, logger)
results = optimizer.run_adam(steps=100, lr=0.01)
```

### Force Finite Differences (for comparison)
```yaml
# config.yaml
optimization:
  gradient_mode: finite_difference
```

```python
# Or via code
config['GRADIENT_MODE'] = 'finite_difference'
optimizer = HBVModelOptimizer(config, logger)
results = optimizer.run_adam(steps=100, lr=0.01)
```

### Require Native Gradients
```yaml
optimization:
  gradient_mode: native  # Will error if model doesn't support autodiff
```

---

## Expected Performance

| Model | Parameters | FD Cost/Step | Native Cost/Step | Speedup |
|-------|------------|--------------|------------------|---------|
| HBV   | 14         | 29 evals     | ~2 evals         | ~15x    |
| LSTM* | 50+        | 101 evals    | ~2 evals         | ~50x    |
| GNN*  | 100+       | 201 evals    | ~2 evals         | ~100x   |

*Future: LSTM/GNN could also implement native gradients via PyTorch autodiff.

---

## Extending to Other Models

To add native gradient support to another model:

1. Implement `supports_native_gradients()` returning `True`
2. Implement `compute_gradient(params, metric)` returning gradient dict
3. Implement `evaluate_with_gradient(params, metric)` for efficiency

For PyTorch models (LSTM, GNN):
```python
def evaluate_with_gradient(self, params, metric='kge'):
    # Enable gradients
    for p in self.model.parameters():
        p.requires_grad_(True)

    # Forward pass
    loss = self._compute_loss(params, metric)

    # Backward pass
    loss.backward()

    # Extract gradients
    grad_dict = {name: p.grad.numpy() for name, p in self.named_parameters()}

    return loss.item(), grad_dict
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Gradient mismatch (FD vs native) | Add validation test comparing both methods |
| Normalized vs physical space confusion | Clear documentation, gradient chain rule in callback |
| JAX not installed | Graceful fallback to FD with warning |
| Numerical issues with autodiff | Gradient clipping applies to both methods |

---

## Summary

This design:
- Keeps existing FD behavior as default
- Allows models to opt-in to native gradients
- Provides config option to force either method
- Enables fair comparison between approaches
- Minimal changes to existing code
- ~15x speedup for HBV, extensible to LSTM/GNN
