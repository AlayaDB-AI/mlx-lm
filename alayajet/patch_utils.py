import importlib
import mlx.nn as nn
from typing import Callable, Dict

_ORIGINAL_METHODS: Dict[str, Callable] = {}

def _get_method_key(cls, method_name):
    return f"{cls.__module__}.{cls.__name__}.{method_name}"

def patch_model(model: nn.Module, attention_pre_hook: Callable = None, attention_post_hook: Callable = None, model_pre_hook: Callable = None):
    attention_cls = None
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
        
    if layers and len(layers) > 0:
        first_layer = layers[0]
        if hasattr(first_layer, "self_attn"):
            attention_cls = type(first_layer.self_attn)
    
    if attention_cls:
        _patch_class_method(attention_cls, "__call__", pre_hook=attention_pre_hook, post_hook=attention_post_hook)
    
    _patch_class_method(type(model), "__call__", pre_hook=model_pre_hook)
    
    if hasattr(model, "model"):
        internal_model = model.model
        if isinstance(internal_model, nn.Module):
             _patch_class_method(type(internal_model), "__call__", pre_hook=model_pre_hook)

def _patch_class_method(cls, method_name, pre_hook=None, post_hook=None):
    if not hasattr(cls, method_name):
        return

    method_key = _get_method_key(cls, method_name)
    if method_key in _ORIGINAL_METHODS:
        return

    original_method = getattr(cls, method_name)
    _ORIGINAL_METHODS[method_key] = original_method

    def make_wrapper(orig_method):
        def wrapper(self, *args, **kwargs):
            if pre_hook:
                res = pre_hook(self, *args, **kwargs)
                if res is not None and isinstance(res, tuple):
                    if len(res) == len(args):
                        args = res
            
            output = orig_method(self, *args, **kwargs)
            
            if post_hook:
                output = post_hook(self, output, *args, **kwargs)
            
            return output
        return wrapper

    setattr(cls, method_name, make_wrapper(original_method))

def patch_class_property(cls, prop_name, new_getter):
    """Replace a class property with a new getter function."""
    if not hasattr(cls, prop_name):
        return

    method_key = _get_method_key(cls, prop_name)
    if method_key in _ORIGINAL_METHODS:
        return

    original_prop = getattr(cls, prop_name)
    _ORIGINAL_METHODS[method_key] = original_prop
    setattr(cls, prop_name, property(new_getter))

def restore_all():
    for key, original_method in _ORIGINAL_METHODS.items():
        parts = key.rsplit(".", 2) 
        if len(parts) != 3:
            continue
            
        mod_name, cls_name, method_name = parts
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            setattr(cls, method_name, original_method)
        except Exception as e:
            print(f"[PatchUtils] Failed to restore {key}: {e}")
    
    _ORIGINAL_METHODS.clear()
