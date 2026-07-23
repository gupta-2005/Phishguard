"""
generate_dataset.py
--------------------
Generates a labeled synthetic dataset of "legit" and "phishing" style URLs.

IMPORTANT NOTE (read before using in production):
This environment has no internet access, so we cannot download a real-world
dataset such as PhishTank or UCI's "Phishing Websites" dataset. Instead, this
script *programmatically constructs* URLs that follow well-documented
patterns from phishing research:
  - legit URLs: short, clean domains, no suspicious keywords, HTTPS, no IPs
  - phishing URLs: brand impersonation, IP hosts, suspicious keywords,
    URL shorteners, excessive hyphens/subdomains, high entropy strings, etc.

For a real deployment, swap this out for a real labeled dataset (e.g.
PhishTank feed, OpenPhish, or the UCI Phishing Websites dataset) and re-run
train_model.py -- the rest of the pipeline (feature_extraction.py,
train_model.py, app.py) does not need to change.
"""

import random
import string
import pandas as pd
from feature_extraction import extract_features, FEATURE_COLUMNS

random.seed(42)

LEGIT_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "stackoverflow.com",
    "amazon.com", "microsoft.com", "apple.com", "nytimes.com", "bbc.com",
    "linkedin.com", "reddit.com", "dropbox.com", "spotify.com", "netflix.com",
    "python.org", "mozilla.org", "cloudflare.com", "wordpress.com",
    "medium.com", "khanacademy.org", "coursera.org", "nasa.gov", "who.int",
    "un.org", "harvard.edu", "mit.edu", "adobe.com", "salesforce.com",
    "shopify.com", "zoom.us", "slack.com", "notion.so", "figma.com",
]

LEGIT_PATHS = [
    "", "/", "/about", "/products", "/blog/2024/annual-report",
    "/docs/getting-started", "/careers", "/contact", "/news/latest",
    "/help/faq", "/pricing", "/download", "/search?q=weather",
    "/user/settings", "/articles/technology-trends",
]

PHISHING_BRANDS = [
    "paypal", "apple", "amazon", "microsoft", "netflix", "bankofamerica",
    "wellsfargo", "chase", "instagram", "facebook", "outlook", "dropbox",
    "linkedin", "americanexpress", "irs", "usps", "dhl", "fedex",
]

PHISHING_SUFFIX_DOMAINS = [
    "secure-login-verify.com", "account-update-center.tk", "verify-now.info",
    "security-check.top", "customer-support-help.xyz", "signin-alert.ga",
    "confirm-identity.cf", "web-security-update.ml", "login-portal.cn",
    "auth-service-check.gq",
]

SUSPICIOUS_PATH_WORDS = [
    "login", "verify", "secure", "update", "confirm", "signin",
    "account-locked", "webscr", "unlock-account", "reset-password",
]

SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cutt.ly"]


def _rand_str(n):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def make_legit_url():
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    use_www = random.random() < 0.5
    host = f"www.{domain}" if use_www else domain
    # Add a touch of variability (query params / ids) so we can generate
    # many unique rows without looping forever over a small combinatorial space.
    if random.random() < 0.4:
        path = f"{path}?ref={_rand_str(5)}" if "?" not in path else path
    elif random.random() < 0.3:
        path = f"{path}/{_rand_str(4)}"
    return f"https://{host}{path}"


def make_phishing_url():
    style = random.choice(["ip_host", "brand_subdomain", "typo_domain",
                            "shortened", "suspicious_tld"])

    if style == "ip_host":
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        path = "/" + random.choice(SUSPICIOUS_PATH_WORDS)
        return f"http://{ip}{path}/{_rand_str(6)}"

    if style == "brand_subdomain":
        brand = random.choice(PHISHING_BRANDS)
        suffix = random.choice(PHISHING_SUFFIX_DOMAINS)
        path = "/" + random.choice(SUSPICIOUS_PATH_WORDS)
        return f"http://{brand}.{suffix}{path}?session={_rand_str(8)}"

    if style == "typo_domain":
        brand = random.choice(PHISHING_BRANDS)
        typo = brand + random.choice(["-secure", "-online", "0", "-verify", "1"])
        tld = random.choice([".com", ".net", ".info", ".top"])
        path = "/" + random.choice(SUSPICIOUS_PATH_WORDS)
        return f"http://{typo}{tld}{path}"

    if style == "shortened":
        service = random.choice(SHORTENERS)
        return f"http://{service}/{_rand_str(7)}"

    # suspicious_tld
    brand = random.choice(PHISHING_BRANDS)
    tld = random.choice([".tk", ".ga", ".ml", ".cf", ".gq", ".xyz", ".top"])
    path = "/" + random.choice(SUSPICIOUS_PATH_WORDS) + "-" + _rand_str(4)
    return f"http://{brand}-account{tld}{path}"


def generate_dataset(n_legit=1200, n_phishing=1200):
    rows = []
    seen = set()

    while len([r for r in rows if r["label"] == 0]) < n_legit:
        url = make_legit_url()
        if url in seen:
            continue
        seen.add(url)
        feats = extract_features(url)
        feats["url"] = url
        feats["label"] = 0  # 0 = safe
        rows.append(feats)

    while len([r for r in rows if r["label"] == 1]) < n_phishing:
        url = make_phishing_url()
        if url in seen:
            continue
        seen.add(url)
        feats = extract_features(url)
        feats["url"] = url
        feats["label"] = 1  # 1 = phishing
        rows.append(feats)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    cols = ["url"] + FEATURE_COLUMNS + ["label"]
    return df[cols]


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/urls_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["label"].value_counts())
    print(df.head())