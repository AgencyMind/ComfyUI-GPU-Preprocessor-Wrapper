import comfy.model_management as model_management
import torch
import logging

# Set up logging
logger = logging.getLogger(__name__)

class MultiGPUPreprocessorWrapper:
    """
    Base wrapper class that temporarily overrides device placement during preprocessor model loading
    AND execution to prevent multi-GPU device conflicts in ControlNet preprocessors.
    
    The problem: Preprocessors auto-download models and load them using model_management.get_torch_device(),
    but ComfyUI-MultiGPU monkey-patches this function with dynamic device assignment, causing model 
    components to split across devices and trigger "Expected all tensors to be on the same device" errors.
    
    The solution: Override get_torch_device() to return consistent device (cuda:0) during
    the ENTIRE preprocessor execution (loading + inference), then restore normal MultiGPU behavior.
    """
    
    def __init__(self, preprocessor_class, target_device='cuda:0'):
        self.preprocessor_class = preprocessor_class
        self.target_device = target_device
        self._preprocessor_instance = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # Must dynamically return the wrapped preprocessor's INPUT_TYPES
        # This will be overridden in specific wrapper subclasses
        if hasattr(cls, 'preprocessor_class'):
            return cls.preprocessor_class.INPUT_TYPES()
        else:
            # Fallback for base class - should not be used directly
            return {"required": {}}
    
    RETURN_TYPES = ("IMAGE",)  # Most ControlNet preprocessors return IMAGE
    FUNCTION = "execute"
    CATEGORY = "preprocessors/gpu_wrapper"
    
    def _ensure_tensors_on_device(self, obj, device):
        """
        Recursively move tensors in nested data structures to the specified device.
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._ensure_tensors_on_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._ensure_tensors_on_device(item, device) for item in obj)
        else:
            return obj
    
    def execute(self, **kwargs):
        """
        Execute preprocessor with extended device override to prevent multi-GPU conflicts.
        Now covers both model loading AND inference execution.
        """
        # Critical: Save original function
        original_get_device = model_management.get_torch_device
        
        try:
            # Force clear any cached models that might have wrong device assignments
            if hasattr(model_management, 'soft_empty_cache'):
                model_management.soft_empty_cache()
            
            # Clear preprocessor instance cache to force fresh loading
            self._preprocessor_instance = None
            # Override with consistent device for ENTIRE execution
            target_torch_device = torch.device(self.target_device)
            model_management.get_torch_device = lambda: target_torch_device
            
            # Ensure all input tensors are on the target device
            kwargs = self._ensure_tensors_on_device(kwargs, target_torch_device)
            
            # Create preprocessor instance if not cached, or reuse existing
            if self._preprocessor_instance is None:
                self._preprocessor_instance = self.preprocessor_class()
            
            # Execute with consistent device context
            function_name = getattr(self.preprocessor_class, 'FUNCTION', 'execute')
            result = getattr(self._preprocessor_instance, function_name)(**kwargs)
            
            # Ensure output tensors are properly placed
            result = self._ensure_tensors_on_device(result, target_torch_device)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in MultiGPUPreprocessorWrapper execution: {e}")
            logger.error(f"Target device: {self.target_device}")
            logger.error(f"Input tensor devices: {[v.device if isinstance(v, torch.Tensor) else type(v) for v in kwargs.values()]}")
            raise
            
        finally:
            # ALWAYS restore original function, even on exception
            model_management.get_torch_device = original_get_device


# Import and create specific wrapper instances with error handling

# DepthAnything V2 Wrapper
try:
    from comfyui_controlnet_aux import NODE_CLASS_MAPPINGS as AUX_NODE_MAPPINGS
    DepthAnythingV2Preprocessor = AUX_NODE_MAPPINGS["DepthAnythingV2Preprocessor"]
    
    class DepthAnythingV2Wrapper(MultiGPUPreprocessorWrapper):
        def __init__(self):
            super().__init__(DepthAnythingV2Preprocessor, 'cuda:0')
        
        @classmethod
        def INPUT_TYPES(cls):
            return DepthAnythingV2Preprocessor.INPUT_TYPES()
            
        RETURN_TYPES = DepthAnythingV2Preprocessor.RETURN_TYPES
        CATEGORY = "preprocessors/gpu_wrapper"
        
    logger.info("DepthAnythingV2Wrapper loaded successfully")
    
except (ImportError, KeyError) as e:
    logger.warning(f"DepthAnythingV2Preprocessor not available: {e}")
    DepthAnythingV2Wrapper = None


# DWPose Wrapper
try:
    if 'AUX_NODE_MAPPINGS' not in locals():
        from comfyui_controlnet_aux import NODE_CLASS_MAPPINGS as AUX_NODE_MAPPINGS
    DWPreprocessor = AUX_NODE_MAPPINGS["DWPreprocessor"]
    
    class DWPreprocessorWrapper(MultiGPUPreprocessorWrapper):
        def __init__(self):
            super().__init__(DWPreprocessor, 'cuda:0')
        
        @classmethod
        def INPUT_TYPES(cls):
            return DWPreprocessor.INPUT_TYPES()
            
        RETURN_TYPES = DWPreprocessor.RETURN_TYPES
        CATEGORY = "preprocessors/gpu_wrapper"
        
    logger.info("DWPreprocessorWrapper loaded successfully")
    
except (ImportError, KeyError) as e:
    logger.warning(f"DWPreprocessor not available: {e}")
    DWPreprocessorWrapper = None


# Canny Edge Wrapper
try:
    if 'AUX_NODE_MAPPINGS' not in locals():
        from comfyui_controlnet_aux import NODE_CLASS_MAPPINGS as AUX_NODE_MAPPINGS
    CannyEdgePreprocessor = AUX_NODE_MAPPINGS["CannyEdgePreprocessor"]
    
    class CannyEdgePreprocessorWrapper(MultiGPUPreprocessorWrapper):
        def __init__(self):
            super().__init__(CannyEdgePreprocessor, 'cuda:0')
        
        @classmethod
        def INPUT_TYPES(cls):
            return CannyEdgePreprocessor.INPUT_TYPES()
            
        RETURN_TYPES = CannyEdgePreprocessor.RETURN_TYPES
        CATEGORY = "preprocessors/gpu_wrapper"
        
    logger.info("CannyEdgePreprocessorWrapper loaded successfully")
    
except (ImportError, KeyError) as e:
    logger.warning(f"CannyEdgePreprocessor not available: {e}")
    CannyEdgePreprocessorWrapper = None


# OpenPose Wrapper
try:
    if 'AUX_NODE_MAPPINGS' not in locals():
        from comfyui_controlnet_aux import NODE_CLASS_MAPPINGS as AUX_NODE_MAPPINGS
    OpenposePreprocessor = AUX_NODE_MAPPINGS["OpenposePreprocessor"]
    
    class OpenposePreprocessorWrapper(MultiGPUPreprocessorWrapper):
        def __init__(self):
            super().__init__(OpenposePreprocessor, 'cuda:0')
        
        @classmethod
        def INPUT_TYPES(cls):
            return OpenposePreprocessor.INPUT_TYPES()
            
        RETURN_TYPES = OpenposePreprocessor.RETURN_TYPES
        CATEGORY = "preprocessors/gpu_wrapper"
        
    logger.info("OpenposePreprocessorWrapper loaded successfully")
    
except (ImportError, KeyError) as e:
    logger.warning(f"OpenposePreprocessor not available: {e}")
    OpenposePreprocessorWrapper = None


# Midas Depth Map Wrapper
try:
    if 'AUX_NODE_MAPPINGS' not in locals():
        from comfyui_controlnet_aux import NODE_CLASS_MAPPINGS as AUX_NODE_MAPPINGS
    MidasDepthMapPreprocessor = AUX_NODE_MAPPINGS["MiDaS-DepthMapPreprocessor"]
    
    class MidasDepthMapWrapper(MultiGPUPreprocessorWrapper):
        def __init__(self):
            super().__init__(MidasDepthMapPreprocessor, 'cuda:0')
        
        @classmethod
        def INPUT_TYPES(cls):
            return MidasDepthMapPreprocessor.INPUT_TYPES()
            
        RETURN_TYPES = MidasDepthMapPreprocessor.RETURN_TYPES
        CATEGORY = "preprocessors/gpu_wrapper"
        
    logger.info("MidasDepthMapWrapper loaded successfully")
    
except (ImportError, KeyError) as e:
    logger.warning(f"MidasDepthMapPreprocessor not available: {e}")
    MidasDepthMapWrapper = None


# Registration dictionaries
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Only register wrappers for available preprocessors
if DepthAnythingV2Wrapper:
    NODE_CLASS_MAPPINGS["DepthAnythingV2Wrapper"] = DepthAnythingV2Wrapper
    NODE_DISPLAY_NAME_MAPPINGS["DepthAnythingV2Wrapper"] = "DepthAnything V2 (GPU Wrapper)"

if DWPreprocessorWrapper:
    NODE_CLASS_MAPPINGS["DWPreprocessorWrapper"] = DWPreprocessorWrapper  
    NODE_DISPLAY_NAME_MAPPINGS["DWPreprocessorWrapper"] = "DWPose (GPU Wrapper)"

if CannyEdgePreprocessorWrapper:
    NODE_CLASS_MAPPINGS["CannyEdgePreprocessorWrapper"] = CannyEdgePreprocessorWrapper
    NODE_DISPLAY_NAME_MAPPINGS["CannyEdgePreprocessorWrapper"] = "Canny Edge (GPU Wrapper)"

if OpenposePreprocessorWrapper:
    NODE_CLASS_MAPPINGS["OpenposePreprocessorWrapper"] = OpenposePreprocessorWrapper
    NODE_DISPLAY_NAME_MAPPINGS["OpenposePreprocessorWrapper"] = "OpenPose (GPU Wrapper)"

if MidasDepthMapWrapper:
    NODE_CLASS_MAPPINGS["MidasDepthMapWrapper"] = MidasDepthMapWrapper
    NODE_DISPLAY_NAME_MAPPINGS["MidasDepthMapWrapper"] = "Midas Depth (GPU Wrapper)"

logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} GPU wrapper nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
