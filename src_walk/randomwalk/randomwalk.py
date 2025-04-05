"""
A class to simulate 2D random walks using NumPy vectorization.

The walk consists of discrete steps in random directions (x or y axis)
with configurable step sizes.
"""

import numpy as np

class RandomWalk:
    """Simulates a 2D random walk with controlled step sizes and directions.

    Attributes:
        steps (int): Total number of steps in the walk
        step_values (ndarray): Valid step sizes available for selection.
    """

    def __init__(self, steps: int, size: int) -> None:
        """Initialize random walk parameters.

        Args:
            steps: Total number of steps in the walk
            size: Maximum step magnitude (exclusive). Actual steps will be
                in [-size, -1] ∪ [1, size] when size > 1

        Note:
            The step values exclude 0 to ensure movement at each step.
        """
        self.steps = steps
        # Generate valid step values from -size to size excluding 0
        self.step_values = np.concatenate((
            np.arange(-size, 0, 1),  # Negative steps
            np.arange(1, size+1, 1)   # Positive steps
        ))

    def __call__(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate and return the random walk coordinates.

        Returns:
            tuple: Two ndarrays containing x and y coordinates respectively

        Process:
            1. Randomly choose direction (x=0/y=1) for each step
            2. Randomly select step sizes from valid values
            3. Assign steps to x/y directions using vectorized operations
            4. Calculate cumulative sum for coordinates
        """
        # Generate random directions (0 for x-axis, 1 for y-axis)
        directions = np.random.choice([0, 1], size=self.steps)

        # Randomly select step sizes from precomputed values
        steps = np.random.choice(self.step_values, size=self.steps)

        # Vectorized step assignment using boolean masks
        # Fixed: Changed 'not directions' to 'directions == 0'
        x_steps = np.where(directions == 0, steps, 0)  # X-direction steps
        y_steps = np.where(directions == 1, steps, 0)  # Y-direction steps

        # Calculate cumulative sums for coordinates
        # Fixed: Corrected typo 'cunsum' to 'cumsum'
        x_values = np.cumsum(x_steps)
        y_values = np.cumsum(y_steps)

        # Prepend starting position (0, 0)
        return np.insert(x_values, 0, 0), np.insert(y_values, 0, 0)