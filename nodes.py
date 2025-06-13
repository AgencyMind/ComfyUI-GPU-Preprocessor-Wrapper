import comfy.model_management as model_management
import torch
import logging

# Set up logging
logger = logging.getLogger(__name__)

def get_device_list():
    """Get list of available devices"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    return devices

class MultiGPUPreprocessorWrapper:
    """
    Base wrapper class that temporarily overrides device placement during preprocessor model loading
    and ensures input tensors are on the correct device to prevent multi-GPU device conflicts.
    """
    
    def __init__(self, preprocessor_class):
        self.preprocessor_class = preprocessor_class
        
    @classmethod
    def INPUT_TYPES(cls):
        # Get base INPUT_TYPES from wrapped preprocessor
        if hasattr(cls, 'preprocessor_class'):
            base_inputs = cls.preprocessor_class.INPUT_TYPES()
        else:
            base_inputs = {"required": {}}
        
        # Add device parameter
        devices = get_device_list()
        default_device = "cuda:0" if "cuda:0" in devices else (devices[1] if len(devices) > 1 else devices[0])
        
        # Ensure optional section exists
        if "optional" not in base_inputs:
            base_inputs["optional"] = {}
        
        base_inputs["optional"]["device"] = (devices, {"default": default_device})
        
        return base_inputs
    
    RETURN_TYPES = ("IMAGE",)  # Most ControlNet preprocessors return IMAGE
    FUNCTION = "execute"
    CATEGORY = "preprocessors/gpu_wrapper"
    
    def execute(self, device="cuda:1", **kwargs):
        """
        Execute preprocessor with device override and input tensor movement.
        """
        target_device = torch.device(device)
        
        # Move input tensors to target device
        moved_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                moved_kwargs[key] = value.to(target_device)
                logger.debug(f"Moved input tensor '{key}' from {value.device} to {target_device}")
            else:
                moved_kwargs[key] = value
        
        # Critical: Save original function
        original_get_device = model_management.get_torch_device
        
        try:
            # Temporarily override with consistent device
            model_management.get_torch_device = lambda: target_device
            logger.debug(f"Temporarily overriding get_torch_device() to return: {target_device}")
            
            # Create and execute original preprocessor using its specific function name
            preprocessor = self.preprocessor_class()
            function_name = getattr(self.preprocessor_class, 'FUNCTION', 'execute')
            result = getattr(preprocessor, function_name)(**moved_kwargs)
            
            logger.debug(f"Successfully executed {self.preprocessor_class.__name__} on {target_device}")
            return result
            
        except Exception as e:
            logger.error(f"Error in MultiGPUPreprocessorWrapper execution: {e}")
            raise
            
        finally:
            # ALWAYS restore original function, even on exception
            model_management.get_torch_device = original_get_device
            logger.debug("Restored original get_torch_device()")


# Import and create specific wrapper instances with error handling

# DepthAnything V2 Wrapper
try:
    from comfyui_controlnet_aux import NODE_CLASS_MAPPINGS as AUX_NODE_MAPPINGS
    DepthAnythingV2Preprocessor = AUX_NODE_MAPPINGS["DepthAnythingV2Preprocessor"]
    
    class DepthAnythingV2Wrapper(MultiGPUPreprocessorWrapper):
        preprocessor_class = DepthAnythingV2Preprocessor
        
        def __init__(self):
            super().__init__(DepthAnythingV2Preprocessor)
        
        @classmethod
        def INPUT_TYPES(cls):
            return super().INPUT_TYPES()
            
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
        preprocessor_class = DWPreprocessor
        
        def __init__(self):
            super().__init__(DWPreprocessor)
        
        @classmethod
        def INPUT_TYPES(cls):
            return super().INPUT_TYPES()
            
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
        preprocessor_class = CannyEdgePreprocessor
        
        def __init__(self):
            super().__init__(CannyEdgePreprocessor)
        
        @classmethod
        def INPUT_TYPES(cls):
            return super().INPUT_TYPES()
            
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
        preprocessor_class = OpenposePreprocessor
        
        def __init__(self):
            super().__init__(OpenposePreprocessor)
        
        @classmethod
        def INPUT_TYPES(cls):
            return super().INPUT_TYPES()
            
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
        preprocessor_class = MidasDepthMapPreprocessor
        
        def __init__(self):
            super().__init__(MidasDepthMapPreprocessor)
        
        @classmethod
        def INPUT_TYPES(cls):
            return super().INPUT_TYPES()
            
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