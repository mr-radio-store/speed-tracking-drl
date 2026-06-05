class SpatialAnalyzer:
    """Handles spatial analytics and positioning logic for detected bounding boxes."""
    
    def __init__(self, frame_width: int):
        self.frame_width = frame_width
        self.one_third = frame_width // 3
        self.two_thirds = (frame_width * 2) // 3

    def get_horizontal_position(self, x1: int, x2: int) -> str:
        """Determines if the object center resides in the Left, Center, or Right zone."""
        center_x = (x1 + x2) // 2
        
        if center_x < self.one_third:
            return "Left"
        elif center_x < self.two_thirds:
            return "Center"
        return "Right"