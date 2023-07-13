"""Tests for the utils module
"""
from numpy.testing import assert_almost_equal

from src.utils import GeoPoint, calculate_e2e_cog

SEATTLE_COORDS = GeoPoint(lat=47.6488514, lon=-122.3482221)
NYC_COORDS = GeoPoint(lat=40.749182, lon=-73.9683312)


def test_calculate_e2e_cog_azimuth() -> None:
    """Test the azimuth calculation against known azimuth"""
    fw_azimuth, _ = calculate_e2e_cog(SEATTLE_COORDS, NYC_COORDS)
    assert_almost_equal(fw_azimuth, 83.12, decimal=1)


def test_calculate_e2e_cog_distance() -> None:
    """Test the distance calculation against known distance"""
    _, distance_km = calculate_e2e_cog(SEATTLE_COORDS, NYC_COORDS)
    assert_almost_equal(distance_km, 3877.1671005347353, decimal=1)
