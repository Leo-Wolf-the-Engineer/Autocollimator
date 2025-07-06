class AccurateGaussian:
    def __init__(self, sigma: float, radius: int):
        self.sigma = sigma
        self.radius = radius
        self.kernel = self._generate_kernel()

    def _generate_kernel(self) -> np.ndarray:
        kernel = np.zeros((2 * self.radius + 1, 2 * self.radius + 1))
        for i in range(-self.radius, self.radius + 1):
            for j in range(-self.radius, self.radius + 1):
                kernel[i + self.radius, j + self.radius] = self._gaussian(i, j)
        return kernel / np.sum(kernel)

    def _gaussian(self, x: int, y: int) -> float:
        return np.exp(-((x ** 2 + y ** 2) / (2 * self.sigma ** 2)))

    def process(self, image: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, self.kernel)

    def process_3d(self, image: np.ndarray) -> np.ndarray:
        return np.stack([self.process(image[:, :, i]) for i in range(image.shape[2])], axis=2)