from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class CarDimensions(ABC):
    @property
    @abstractmethod
    def distance_back_to_front_wheel(self) -> float:
        """
        The distance between the back and front wheel (i.e. the L value for he bicycle model)
        """
        pass

    @property
    @abstractmethod
    def center_point_offset(self) -> Tuple[float, float]:
        """
        The position of the center point of the car, relative to the anchor point.
        """
        pass

    @property
    @abstractmethod
    def bounding_box_size(self) -> Tuple[float, float]:
        """
        If the car can be approximated using a rectangle in 2D space (i.e. a bounding box),
        these are the bounds of such approximation (i.e. width and length)
        """
        pass

    @property
    @abstractmethod
    def radius(self) -> float:
        """
        If the car can be approximated using two circles in 2D space,
        this is the radius of each of the circles.
        """
        pass

    @property
    @abstractmethod
    def circle_centers(self) -> np.ndarray:
        """
        If the car can be approximated using circles in 2D space,
        these are the offsets of the circle centers from the anchor point of the vehicle.
        """
        pass


class _SimpleBackWheelAnchoredCarDimensions(CarDimensions, ABC):
    def __init__(self, skip_back_circle_collision_checking=False):
        self.skip_back_circle_collision_checking = skip_back_circle_collision_checking

    @property
    def center_point_offset(self) -> Tuple[float, float]:
        # the anchor point is the back wheel -> the center point is at the L / 2 distance.
        return self.distance_back_to_front_wheel / 2, 0.0

    @property
    def radius(self) -> float:
        width, _ = self.bounding_box_size
        return width / (2 ** .5)

    @property
    def circle_centers(self) -> np.ndarray:
        width, length = self.bounding_box_size
        offset = length / 2 - width / 2

        c_off_y, c_off_x = self.center_point_offset

        if self.skip_back_circle_collision_checking:
            return np.array([[c_off_y + offset, c_off_x]])
        else:
            return np.array([
                [c_off_y + offset, c_off_x],
                [c_off_y - offset, c_off_x],
            ])


class BicycleModelDimensions(_SimpleBackWheelAnchoredCarDimensions):
    @property
    def distance_back_to_front_wheel(self) -> float:
        return 2.86

    @property
    def bounding_box_size(self) -> Tuple[float, float]:
        # an arbitrary width + some extra margin at the back and the front
        return 2.0, self.distance_back_to_front_wheel + 0.64

class BicycleRealDimensions(_SimpleBackWheelAnchoredCarDimensions):
    @property
    def distance_back_to_front_wheel(self) -> float:
        return 1.0

    @property
    def bounding_box_size(self) -> Tuple[float, float]:
        # an arbitrary width + some extra margin at the back and the front
        return 0.45, self.distance_back_to_front_wheel + 0.64

class PriusDimensions(_SimpleBackWheelAnchoredCarDimensions):

    def __init__(self, scaling_factor: float = 1., skip_back_circle_collision_checking=False):
        super().__init__(skip_back_circle_collision_checking)
        self._scaling_factor = scaling_factor

    @property
    def distance_back_to_front_wheel(self) -> float:
        # determine better, if we use the Prius
        return 4 * self._scaling_factor

    @property
    def bounding_box_size(self) -> Tuple[float, float]:
        # this was determined empirically by simulating and placing obstacles around the prius
        return 2.04 * self._scaling_factor, 4.84 * self._scaling_factor
