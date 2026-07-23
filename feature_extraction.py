"""
feature_extraction.py
----------------------
Extracts lexical / structural features from a URL that are commonly used
in academic and industry phishing-detection literature (e.g. presence of
an IP address instead of a domain, use of URL shorteners, count of
suspicious characters, domain age proxies, etc).

No network calls are made here that could be abused -- everything is
derived purely from the URL *string* itself (fast, safe, works offline).
"""

import re
import math
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "signin", "sign-in", "webscr", "ebayisapi", "paypal",
    "suspend", "alert", "wallet", "invoice", "unlock", "recover",
    "password", "security", "authenticat",
]

SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bitly.com", "cutt.ly", "rb.gy", "shorte.st",
]

BRAND_NAMES = [
    "paypal", "apple", "amazon", "microsoft", "google", "facebook",
    "netflix", "bankofamerica", "wellsfargo", "chase", "instagram",
    "whatsapp", "outlook", "dropbox", "linkedin",
]


def _shannon_entropy(s: str) -> float:
    """Higher entropy often correlates with randomly generated / obfuscated domains."""
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def _has_ip_address(hostname: str) -> int:
    ipv4 = r"^(?:\d{1,3}\.){3}\d{1,3}$"
    ipv4_hex = r"^0x[0-9a-fA-F]+"  # obfuscated hex/octal IPs
    if hostname is None:
        return 0
    if re.match(ipv4, hostname) or re.match(ipv4_hex, hostname):
        return 1
    return 0


def _brand_in_subdomain_or_path(hostname: str, path: str) -> int:
    """Detects brand name stuffed into subdomain/path while not being the real
    registered domain -- a classic phishing trick, e.g. paypal.security-check.com"""
    hostname = hostname or ""
    path = path or ""
    parts = hostname.split(".")
    registrable = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
    for brand in BRAND_NAMES:
        if brand in hostname and brand not in registrable:
            return 1
        if brand in path.lower():
            return 1
    return 0


def extract_features(url: str) -> dict:
    """Extract a dictionary of numeric features from a single URL string."""
    url = (url or "").strip()

    # Ensure a scheme is present so urlparse behaves consistently
    parse_target = url if re.match(r"^[a-zA-Z]+://", url) else "http://" + url
    parsed = urlparse(parse_target)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    features = {}

    # --- Basic length-based features ---
    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)

    # --- Character composition ---
    features["count_dots"] = url.count(".")
    features["count_hyphens"] = url.count("-")
    features["count_at"] = url.count("@")
    features["count_question"] = url.count("?")
    features["count_equal"] = url.count("=")
    features["count_underscore"] = url.count("_")
    features["count_ampersand"] = url.count("&")
    features["count_percent"] = url.count("%")
    features["count_digits"] = sum(c.isdigit() for c in url)
    features["count_slashes"] = url.count("/")
    features["count_www"] = url.lower().count("www")

    # --- Structural indicators ---
    features["has_ip_address"] = _has_ip_address(hostname)
    features["uses_https"] = 1 if parsed.scheme == "https" else 0
    features["has_port"] = 1 if parsed.port not in (None, 80, 443) else 0
    features["subdomain_count"] = max(hostname.count(".") - 1, 0)

    # --- Suspicious content ---
    lowered = url.lower()
    features["has_suspicious_word"] = int(
        any(word in lowered for word in SUSPICIOUS_WORDS)
    )
    features["is_shortened"] = int(
        any(service in lowered for service in SHORTENING_SERVICES)
    )
    features["brand_impersonation"] = _brand_in_subdomain_or_path(hostname, path)

    # --- Entropy (randomness) of hostname ---
    features["hostname_entropy"] = round(_shannon_entropy(hostname), 3)

    # --- Ratio features ---
    features["digit_ratio"] = round(
        features["count_digits"] / max(len(url), 1), 3
    )

    # --- Query string complexity ---
    features["query_length"] = len(query)
    features["has_query"] = 1 if query else 0

    return features


# The exact column order the model was trained on. Keeping this centralized
# means feature extraction and model training/inference always stay in sync.
FEATURE_COLUMNS = [
    "url_length", "hostname_length", "path_length",
    "count_dots", "count_hyphens", "count_at", "count_question",
    "count_equal", "count_underscore", "count_ampersand", "count_percent",
    "count_digits", "count_slashes", "count_www",
    "has_ip_address", "uses_https", "has_port", "subdomain_count",
    "has_suspicious_word", "is_shortened", "brand_impersonation",
    "hostname_entropy", "digit_ratio", "query_length", "has_query",
]


def extract_feature_vector(url: str):
    """Return features as an ordered list matching FEATURE_COLUMNS."""
    feats = extract_features(url)
    return [feats[col] for col in FEATURE_COLUMNS]


if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login/verify-account",
        "http://paypal.com.security-update-login.tk/webscr?cmd=login",
        "https://bit.ly/3xAbCde",
    ]
    for u in test_urls:
        print(u)
        print(extract_features(u))
        print()