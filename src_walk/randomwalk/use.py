"""
Module for generating multiple random walk results using the RandomWalk class.

Provides a generator interface for efficient memory usage when processing
large numbers of walks.
"""

from collections import namedtuple
from typing import Generator
import randomwalk as rw
import tqdm

# Define a lightweight structure to store coordinate results
Result = namedtuple("Result", ["x_coords", "y_coords"])  # Fixed comma position


def use_randomwalk(total_walks: int,
                   walk_step: int,
                   size: int) -> Generator(Result, Result, ...):
    """Generate multiple random walk results through a memory-efficient generator.

    Args:
        total_walks: Number of complete walks to generate
        walk_step: Number of steps per individual walk
        size: Maximum step magnitude for the random walk (exclusive)

    Yields:
        Result: Named tuple containing x and y coordinates for each walk

    Example:
        >>> for walk in use_randomwalk(5, 1000, 3):
        ...     print(walk.x_coords.shape, walk.y_coords.shape)
        (1001,) (1001,)

    Note:
        Uses generator pattern to handle large numbers of walks efficiently
    """
    # Initialize random walk generator with specified parameters
    randomwalk = rw.RandomWalk(walk_step, size)

    # Generate requested number of walks
    for _ in tqdm.tqdm(range(total_walks),
                       desc="walk gernerating...",
                       unit="walk",
                       colour="black"):
        # Yield individual walk results while maintaining generator state
        yield Result(*randomwalk())