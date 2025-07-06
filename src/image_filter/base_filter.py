# image_filter/base_filter.py
class BaseFilter:
    """Base class for all image filters"""

    def apply(self, image):
        """Apply the filter to the image"""
        raise NotImplementedError("Subclasses must implement apply()")

    def get_parameters(self):
        """Get the current parameters"""
        return self.__dict__

    def set_parameters(self, **kwargs):
        """Update filter parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)