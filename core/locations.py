import math

WARD_METRICS = {
    "Ward 12 - Shivajinagar": {"city_code": "PMC", "zone_code": "SHIV", "pin": "411005", "lat": 18.5314, "lng": 73.8446},
    "Ward 08 - Kothrud": {"city_code": "PMC", "zone_code": "KOTR", "pin": "411038", "lat": 18.5074, "lng": 73.8077},
    "Ward 14 - Baner / Balewadi": {"city_code": "PMC", "zone_code": "BANR", "pin": "411045", "lat": 18.5590, "lng": 73.7868},
    "Ward 04 - Hadapsar": {"city_code": "PMC", "zone_code": "HADP", "pin": "411028", "lat": 18.5089, "lng": 73.9259},
    "Ward 02 - Pimpri": {"city_code": "PCMC", "zone_code": "PIMP", "pin": "411018", "lat": 18.6298, "lng": 73.7997}
}

def calculate_geofence_distance(lat1, lon1, lat2, lon2):
    """Calculates geodesic distance in meters using the Haversine formula."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c