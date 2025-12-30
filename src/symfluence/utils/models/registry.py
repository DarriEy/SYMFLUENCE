"""
SYMFLUENCE Model Registry

Provides a central registry for hydrological models, preprocessors, 
runners, and postprocessors to enable easy extension.
"""
import logging

class ModelRegistry:
    _preprocessors = {}
    _runners = {}
    _postprocessors = {}
    _runner_methods = {}

    @classmethod
    def register_preprocessor(cls, model_name):
        def decorator(preprocessor_cls):
            cls._preprocessors[model_name] = preprocessor_cls
            return preprocessor_cls
        return decorator

    @classmethod
    def register_runner(cls, model_name, method_name="run"):
        def decorator(runner_cls):
            cls._runners[model_name] = runner_cls
            cls._runner_methods[model_name] = method_name
            return runner_cls
        return decorator

    @classmethod
    def register_postprocessor(cls, model_name):
        def decorator(postprocessor_cls):
            cls._postprocessors[model_name] = postprocessor_cls
            return postprocessor_cls
        return decorator

    @classmethod
    def get_preprocessor(cls, model_name):
        return cls._preprocessors.get(model_name)

    @classmethod
    def get_runner(cls, model_name):
        return cls._runners.get(model_name)

    @classmethod
    def get_postprocessor(cls, model_name):
        return cls._postprocessors.get(model_name)

    @classmethod
    def get_runner_method(cls, model_name):
        return cls._runner_methods.get(model_name, "run")

    @classmethod
    def list_models(cls):
        return sorted(list(set(cls._runners.keys()) | set(cls._preprocessors.keys())))
