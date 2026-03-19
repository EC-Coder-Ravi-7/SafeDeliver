"""
SafeDeliver — Fraud Risk Scoring Engine
========================================
Detects GPS spoofing and coordinated fraud rings
among delivery partners claiming parametric insurance payouts.

Author: SafeDeliver Team
Hackathon: DEVTrails 2026 — Guidewire University Hackathon
"""

# ─────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────
import math
import random
from dataclasses import dataclass
from typing import List, Optional


# ─────────────────────────────────────────
#  DATA MODEL
#  Represents one claim event from a delivery partner
# ─────────────────────────────────────────
@dataclass
class ClaimEvent:
    partner_id: str
    partner_name: str

    # Location signals
    gps_lat: float
    gps_lon: float
    ip_lat: float               # IP-based geolocation latitude
    ip_lon: float               # IP-based geolocation longitude

    # Device sensor signals
    accelerometer_variance: float   # Low = phone not moving (suspicious)
    battery_drain_rate: float       # %/hour — high drain = likely outdoors
    network_type: str               # "wifi" | "4g" | "3g" | "2g"
    signal_strength_dbm: int        # Lower (more negative) = weaker signal

    # Context signals
    weather_storm_confirmed: bool   # Did weather API confirm storm nearby?
    active_order: bool              # Does partner have an active delivery order?
    nearby_partners_claiming: int   # How many partners claiming distress in same zone?

    # NLP signal
    claim_text_similarity_score: float  # 0-1: similarity to other claims (high = scripted)

    # History
    prior_fraud_flags: int          # Number of past fraud flags on this account


# ─────────────────────────────────────────
#  SCORING WEIGHTS
#  Each signal contributes points to the Fraud Risk Score (0-100)
# ─────────────────────────────────────────
WEIGHTS = {
    "gps_ip_mismatch":          25,   # GPS location vs IP location mismatch
    "no_motion":                20,   # Accelerometer shows phone is completely still
    "home_wifi":                10,   # Connected to Wi-Fi (not cellular) during claimed distress
    "no_weather_confirmation":  10,   # Weather API finds no storm at claimed location
    "no_active_order":           8,   # No active delivery order during claimed distress
    "fraud_ring_cluster":       15,   # Many partners claiming from same zone simultaneously
    "scripted_nlp":             10,   # Claim text matches others (copy-paste fraud)
    "prior_flags":               5,   # Partner has prior fraud history
    # Max total = 103 (capped at 100)
}


# ─────────────────────────────────────────
#  HELPER: Calculate distance between two GPS coordinates (km)
# ─────────────────────────────────────────
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─────────────────────────────────────────
#  CORE SCORING ENGINE
# ─────────────────────────────────────────
def compute_fraud_risk_score(event: ClaimEvent) -> dict:
    """
    Analyzes a claim event and returns a Fraud Risk Score (0-100)
    along with a breakdown of which signals were triggered.
    """
    score = 0
    signals_triggered = []

    # ── Signal 1: GPS vs IP Location Mismatch ──────────────────────────────
    gps_ip_distance_km = haversine_distance(
        event.gps_lat, event.gps_lon,
        event.ip_lat, event.ip_lon
    )
    if gps_ip_distance_km > 2.0:  # More than 2km apart = suspicious
        score += WEIGHTS["gps_ip_mismatch"]
        signals_triggered.append(
            f"GPS vs IP mismatch: {gps_ip_distance_km:.1f} km apart (+{WEIGHTS['gps_ip_mismatch']} pts)"
        )

    # ── Signal 2: No Motion Detected ──────────────────────────────────────
    # A genuinely stranded worker's phone shows movement/vibration
    # Spoofers sit at home — near-zero accelerometer variance
    if event.accelerometer_variance < 0.05:
        score += WEIGHTS["no_motion"]
        signals_triggered.append(
            f"No motion detected (variance={event.accelerometer_variance}) (+{WEIGHTS['no_motion']} pts)"
        )

    # ── Signal 3: Home Wi-Fi Connected ────────────────────────────────────
    # A stranded worker in a storm/flood zone would be on cellular, not Wi-Fi
    if event.network_type == "wifi":
        score += WEIGHTS["home_wifi"]
        signals_triggered.append(
            f"Connected to Wi-Fi during claimed distress (+{WEIGHTS['home_wifi']} pts)"
        )

    # ── Signal 4: Weather API — No Storm Confirmed ────────────────────────
    if not event.weather_storm_confirmed:
        score += WEIGHTS["no_weather_confirmation"]
        signals_triggered.append(
            f"Weather API: No storm/flood at claimed location (+{WEIGHTS['no_weather_confirmation']} pts)"
        )

    # ── Signal 5: No Active Delivery Order ────────────────────────────────
    if not event.active_order:
        score += WEIGHTS["no_active_order"]
        signals_triggered.append(
            f"No active delivery order at time of claim (+{WEIGHTS['no_active_order']} pts)"
        )

    # ── Signal 6: Fraud Ring — Cluster Detection ──────────────────────────
    # If 5+ partners in same zone all claim distress simultaneously → ring
    if event.nearby_partners_claiming >= 5:
        score += WEIGHTS["fraud_ring_cluster"]
        signals_triggered.append(
            f"Fraud ring: {event.nearby_partners_claiming} partners claiming in same zone "
            f"(+{WEIGHTS['fraud_ring_cluster']} pts)"
        )

    # ── Signal 7: NLP — Scripted Claim Text ───────────────────────────────
    # High similarity to other claims = copy-paste coordinated fraud
    if event.claim_text_similarity_score > 0.75:
        score += WEIGHTS["scripted_nlp"]
        signals_triggered.append(
            f"NLP: Claim text {event.claim_text_similarity_score*100:.0f}% similar to other claims "
            f"(+{WEIGHTS['scripted_nlp']} pts)"
        )

    # ── Signal 8: Prior Fraud History ─────────────────────────────────────
    if event.prior_fraud_flags > 0:
        score += WEIGHTS["prior_flags"]
        signals_triggered.append(
            f"Prior fraud flags: {event.prior_fraud_flags} (+{WEIGHTS['prior_flags']} pts)"
        )

    # Cap score at 100
    score = min(score, 100)

    return {
        "partner_id": event.partner_id,
        "partner_name": event.partner_name,
        "fraud_risk_score": score,
        "tier": get_tier(score),
        "action": get_action(score),
        "signals_triggered": signals_triggered,
        "gps_ip_distance_km": round(gps_ip_distance_km, 2),
    }


# ─────────────────────────────────────────
#  TIER CLASSIFICATION
# ─────────────────────────────────────────
def get_tier(score: int) -> str:
    if score <= 30:   return "🟢 GREEN  — Auto-Approve"
    if score <= 55:   return "🟡 YELLOW — Soft Review"
    if score <= 75:   return "🟠 ORANGE — Hold & Verify"
    return              "🔴 RED    — Fraud Escalated"


def get_action(score: int) -> str:
    if score <= 30:
        return "Payout processed immediately. No friction for worker."
    if score <= 55:
        return "Payout processed. Worker asked for photo upload within 2 hours."
    if score <= 75:
        return "Payout held 6 hours. Human reviewer alerted. Worker contacted via chat."
    return "Payout withheld. Case escalated to fraud team. Worker notified with appeal link."


# ─────────────────────────────────────────
#  FRAUD RING DETECTOR
#  Scans a batch of claims for coordinated cluster patterns
# ─────────────────────────────────────────
def detect_fraud_rings(results: List[dict], threshold: int = 75) -> List[str]:
    """
    Identifies groups of partners with high FRS scores
    submitted in the same time window — likely a coordinated ring.
    """
    high_risk = [r for r in results if r["fraud_risk_score"] >= threshold]
    if len(high_risk) >= 3:
        ids = [r["partner_id"] for r in high_risk]
        return [f"⚠️  FRAUD RING DETECTED: {len(high_risk)} partners flagged → {ids}"]
    return ["✅ No coordinated fraud ring detected in this batch."]


# ─────────────────────────────────────────
#  PRINT RESULT
# ─────────────────────────────────────────
def print_result(result: dict):
    print("=" * 55)
    print(f"  Partner : {result['partner_name']} ({result['partner_id']})")
    print(f"  FRS     : {result['fraud_risk_score']} / 100")
    print(f"  Tier    : {result['tier']}")
    print(f"  Action  : {result['action']}")
    print(f"  GPS↔IP  : {result['gps_ip_distance_km']} km apart")
    if result["signals_triggered"]:
        print("  Signals :")
        for s in result["signals_triggered"]:
            print(f"    → {s}")
    else:
        print("  Signals : None triggered ✅")
    print("=" * 55)


# ─────────────────────────────────────────
#  DEMO: Run sample claims
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("\n🚚 SafeDeliver — Fraud Risk Scoring Engine")
    print("DEVTrails 2026 · Guidewire Hackathon\n")

    # ── Case 1: Genuine stranded worker ───────────────────────────────────
    genuine_worker = ClaimEvent(
        partner_id="P1042",
        partner_name="Ravi Kumar",
        gps_lat=19.0760, gps_lon=72.8777,   # Mumbai
        ip_lat=19.0761,  ip_lon=72.8780,     # Same location (genuine)
        accelerometer_variance=0.82,          # Phone moving (walking in rain)
        battery_drain_rate=18.5,              # High drain (flashlight on)
        network_type="3g",                    # Weak cellular (storm area)
        signal_strength_dbm=-98,              # Poor signal
        weather_storm_confirmed=True,         # Storm confirmed ✅
        active_order=True,                    # Has active delivery ✅
        nearby_partners_claiming=1,           # Only him
        claim_text_similarity_score=0.12,     # Unique description
        prior_fraud_flags=0,
    )

    # ── Case 2: GPS Spoofer at home ────────────────────────────────────────
    spoofer = ClaimEvent(
        partner_id="F001",
        partner_name="Fake GPS #1",
        gps_lat=19.0760, gps_lon=72.8777,   # Faked GPS (Mumbai distress zone)
        ip_lat=19.1234,  ip_lon=72.9100,     # Real IP location (home, 6km away)
        accelerometer_variance=0.01,          # Phone completely still (at home)
        battery_drain_rate=3.2,               # Low drain (idle at home)
        network_type="wifi",                  # Home Wi-Fi (not cellular)
        signal_strength_dbm=-42,              # Strong signal (home router)
        weather_storm_confirmed=False,        # No storm at claimed location
        active_order=False,                   # No active order
        nearby_partners_claiming=12,          # 12 others claiming same zone = ring
        claim_text_similarity_score=0.94,     # Almost identical to other claims
        prior_fraud_flags=2,
    )

    # ── Case 3: Borderline / Uncertain ────────────────────────────────────
    borderline = ClaimEvent(
        partner_id="P0887",
        partner_name="Imran Baig",
        gps_lat=19.0500, gps_lon=72.8500,
        ip_lat=19.0520,  ip_lon=72.8510,
        accelerometer_variance=0.15,
        battery_drain_rate=9.0,
        network_type="4g",
        signal_strength_dbm=-75,
        weather_storm_confirmed=False,        # No storm (raises suspicion)
        active_order=True,
        nearby_partners_claiming=2,
        claim_text_similarity_score=0.45,
        prior_fraud_flags=0,
    )

    # ── Run scoring ────────────────────────────────────────────────────────
    cases = [genuine_worker, spoofer, borderline]
    results = []

    for case in cases:
        result = compute_fraud_risk_score(case)
        results.append(result)
        print_result(result)

    # ── Fraud Ring Detection ───────────────────────────────────────────────
    print("\n🔍 Fraud Ring Analysis:")
    ring_alerts = detect_fraud_rings(results)
    for alert in ring_alerts:
        print(f"  {alert}")

    print("\n✅ Scoring complete.\n")
