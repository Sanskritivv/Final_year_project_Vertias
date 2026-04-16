import cv2
import numpy as np

class IrisProcessor:
    """
    Handles iris localization (pupil and iris detection) and 
    normalization (converting to a rectangular strip).
    """
    
    # Standard parameters for Hough Circles based on UBIRIS dataset recommendations
    PUPIL_PARAMS = {
        "dp": 1.0,
        "minDist": 200,
        "param1": 10,
        "param2": 15,
        "minRadius": 20,
        "maxRadius": 150
    }

    IRIS_PARAMS = {
        "dp": 1.5,
        "minDist": 200,
        "param1": 10,
        "param2": 30,
        "minRadius": 100,
        "maxRadius": 300
    }

    def __init__(self, output_h=224, output_w=400):
        self.output_h = output_h
        self.output_w = output_w

    def process_image_data(self, image_data):
        """
        Main entry point for image data processing.
        :param image_data: numpy array (BGR)
        :return: (is_valid, normalized_iris)
        """
        try:
            # 1. Localize
            iris_circle, pupil_circle = self.localize(image_data)
            if iris_circle is None or pupil_circle is None:
                return False, None
            
            # 2. Normalize
            normalized = self.normalize(image_data, iris_circle, pupil_circle)
            return True, normalized
        except Exception as e:
            print(f"Extraction failed: {e}")
            return False, None

    def localize(self, img):
        """
        Find iris and pupil circles using Hough Transform.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Find Iris
        iris_circles = cv2.HoughCircles(
            gray_enhanced, cv2.HOUGH_GRADIENT,
            **self.IRIS_PARAMS
        )
        
        if iris_circles is None:
            return None, None
        
        # Take the best candidate (usually closest to image center)
        h, w = gray.shape[:2]
        center = np.array([w // 2, h // 2])
        iris = iris_circles[0][np.argmin(np.linalg.norm(iris_circles[0, :, :2] - center, axis=1))]
        
        # Find Pupil (within/around the iris center)
        pupil_circles = cv2.HoughCircles(
            gray_enhanced, cv2.HOUGH_GRADIENT,
            **self.PUPIL_PARAMS
        )
        
        if pupil_circles is None:
            return iris, None
            
        pupil = pupil_circles[0][np.argmin(np.linalg.norm(pupil_circles[0, :, :2] - iris[:2], axis=1))]
        
        return iris, pupil

    def normalize(self, img, iris, pupil):
        """
        Daugman's rubber sheet model implementation.
        """
        ix, iy, ir = iris
        px, py, pr = pupil
        
        nsamples = self.output_w
        nradii = self.output_h
        
        theta = np.linspace(0, 2*np.pi, nsamples)
        
        # Rectangular strip
        polar_array = np.zeros((nradii, nsamples), dtype=np.uint8)
        
        for i in range(nsamples):
            t = theta[i]
            
            # Points on pupil boundary
            p_x = px + pr * np.cos(t)
            p_y = py + pr * np.sin(t)
            
            # Points on iris boundary
            i_x = ix + ir * np.cos(t)
            i_y = iy + ir * np.sin(t)
            
            # Radius range between pupil and iris
            for r in range(nradii):
                alpha = r / nradii
                x = (1 - alpha) * p_x + alpha * i_x
                y = (1 - alpha) * p_y + alpha * i_y
                
                # Sample the image
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    polar_array[r, i] = img[int(y), int(x), 0] # Use one channel for simplicity
                else:
                    polar_array[r, i] = 0
                    
        return polar_array
