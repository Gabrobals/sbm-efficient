"""
Adaptive-K Licensing System

Validates commercial licenses for Adaptive-K SDK.
Community tier (Apache 2.0) is always free and fully functional.
Commercial tiers add support, optimizations, and enterprise features.

Copyright 2026 Vertex Data S.r.l.
Licensed under Apache License 2.0
"""

import hashlib
import json
import base64
import os
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LicenseInfo:
    """License information container."""
    tier: str
    valid: bool
    features: list
    limits: Dict[str, Any]
    company: str = "Community User"
    expires: Optional[str] = None
    message: str = ""


class LicenseValidator:
    """Validates commercial licenses for Adaptive-K.
    
    The Community tier is always available and fully functional.
    Commercial tiers (Professional, Enterprise) require a license key.
    
    Example:
        >>> validator = LicenseValidator()
        >>> info = validator.validate()
        >>> print(info.tier)  # "community"
        
        >>> validator = LicenseValidator(license_key="eyJ...")
        >>> info = validator.validate()
        >>> print(info.tier)  # "professional" or "enterprise"
    """
    
    # Feature sets by tier
    FEATURES = {
        "community": [
            "base_routing",
            "calibration", 
            "cli",
            "basic_stats"
        ],
        "professional": [
            "base_routing",
            "calibration",
            "cli",
            "basic_stats",
            "cuda_kernels",
            "vllm_integration",
            "huggingface_integration",
            "tensorrt_integration",
            "priority_support",
            "no_attribution"
        ],
        "enterprise": [
            "base_routing",
            "calibration",
            "cli",
            "basic_stats",
            "cuda_kernels",
            "vllm_integration",
            "huggingface_integration",
            "tensorrt_integration",
            "priority_support",
            "no_attribution",
            "custom_optimization",
            "dedicated_support",
            "sla_guarantee",
            "redistribution_rights",
            "early_access"
        ]
    }
    
    # Usage limits by tier
    LIMITS = {
        "community": {"requests_per_month": None},  # Unlimited
        "professional": {"requests_per_month": 10_000_000},
        "enterprise": {"requests_per_month": None}  # Unlimited
    }
    
    def __init__(self, license_key: Optional[str] = None):
        """Initialize validator with optional license key.
        
        Args:
            license_key: Commercial license key, or None for community tier.
                        Can also be set via ADAPTIVE_K_LICENSE env var.
        """
        self.license_key = license_key or os.environ.get("ADAPTIVE_K_LICENSE")
        self._cached_info: Optional[LicenseInfo] = None
    
    def validate(self) -> LicenseInfo:
        """Validate license and return tier information.
        
        Returns:
            LicenseInfo with tier, features, limits, and validity status.
        """
        if self._cached_info is not None:
            return self._cached_info
        
        # No key = community tier (always valid)
        if not self.license_key:
            self._cached_info = LicenseInfo(
                tier="community",
                valid=True,
                features=self.FEATURES["community"],
                limits=self.LIMITS["community"],
                message="Community tier - full functionality, Apache 2.0 license"
            )
            return self._cached_info
        
        # Validate commercial license
        try:
            payload = self._decode_key(self.license_key)
            
            # Check expiration
            if self._is_expired(payload):
                self._cached_info = LicenseInfo(
                    tier="expired",
                    valid=False,
                    features=self.FEATURES["community"],
                    limits=self.LIMITS["community"],
                    company=payload.get("company", "Unknown"),
                    expires=payload.get("expires"),
                    message="License expired. Falling back to community tier. Contact amministrazione@vertexdata.it to renew."
                )
                return self._cached_info
            
            tier = payload.get("tier", "community")
            self._cached_info = LicenseInfo(
                tier=tier,
                valid=True,
                features=self.FEATURES.get(tier, self.FEATURES["community"]),
                limits=self.LIMITS.get(tier, self.LIMITS["community"]),
                company=payload.get("company", "Unknown"),
                expires=payload.get("expires"),
                message=f"{tier.capitalize()} license active until {payload.get('expires', 'N/A')}"
            )
            return self._cached_info
            
        except ValueError as e:
            self._cached_info = LicenseInfo(
                tier="invalid",
                valid=False,
                features=self.FEATURES["community"],
                limits=self.LIMITS["community"],
                message=f"Invalid license key: {e}. Using community tier."
            )
            return self._cached_info
    
    def _decode_key(self, key: str) -> Dict[str, Any]:
        """Decode and verify license key.
        
        Key format: base64(json_payload).signature
        
        Args:
            key: License key string
            
        Returns:
            Decoded payload dictionary
            
        Raises:
            ValueError: If key format or signature is invalid
        """
        parts = key.strip().split(".")
        if len(parts) != 2:
            raise ValueError("Invalid key format")
        
        try:
            payload_json = base64.b64decode(parts[0]).decode('utf-8')
            payload = json.loads(payload_json)
        except Exception:
            raise ValueError("Cannot decode payload")
        
        # Verify signature
        signature = parts[1]
        expected_sig = self._compute_signature(parts[0])
        
        if signature != expected_sig:
            raise ValueError("Invalid signature")
        
        # Validate required fields
        if "tier" not in payload or "expires" not in payload:
            raise ValueError("Missing required fields")
        
        return payload
    
    def _compute_signature(self, payload_b64: str) -> str:
        """Compute signature for payload.
        
        Note: In production, use asymmetric crypto (RSA/ECDSA).
        This is a simplified HMAC-like approach.
        """
        # Secret should be in env var, not hardcoded
        secret = os.environ.get("ADAPTIVE_K_SECRET", "VERTEX_ADAPTIVE_K_2026")
        sig_input = f"{payload_b64}{secret}".encode('utf-8')
        return hashlib.sha256(sig_input).hexdigest()[:16]
    
    def _is_expired(self, payload: Dict[str, Any]) -> bool:
        """Check if license has expired."""
        try:
            expires_str = payload.get("expires", "")
            # Support both date and datetime formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    expires = datetime.strptime(expires_str, fmt)
                    return datetime.now() > expires
                except ValueError:
                    continue
            return True  # Invalid date format = expired
        except Exception:
            return True
    
    def has_feature(self, feature: str) -> bool:
        """Check if current license includes a specific feature.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is available
        """
        info = self.validate()
        return feature in info.features
    
    def check_limit(self, limit_name: str, current_value: int) -> bool:
        """Check if usage is within limits.
        
        Args:
            limit_name: Name of the limit to check
            current_value: Current usage value
            
        Returns:
            True if within limits, False if exceeded
        """
        info = self.validate()
        limit = info.limits.get(limit_name)
        if limit is None:
            return True  # No limit
        return current_value <= limit


def generate_license_key(
    company: str,
    tier: str,
    expires: str,
    email: str = "",
    secret: Optional[str] = None
) -> str:
    """Generate a license key (admin tool).
    
    This function should only be used by Vertex Data administrators.
    
    Args:
        company: Company name
        tier: License tier (professional, enterprise)
        expires: Expiration date (YYYY-MM-DD)
        email: Contact email
        secret: Signing secret (uses env var if not provided)
        
    Returns:
        License key string
        
    Example:
        >>> key = generate_license_key(
        ...     company="Acme AI",
        ...     tier="professional",
        ...     expires="2027-01-14",
        ...     email="cto@acme.ai"
        ... )
        >>> print(key)
        eyJjb21wYW55IjoiQWNtZSBBSSIsInRpZXIiOiJwcm9mZXNza...
    """
    if tier not in ["professional", "enterprise"]:
        raise ValueError(f"Invalid tier: {tier}. Must be professional or enterprise.")
    
    # Validate date format
    try:
        datetime.strptime(expires, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {expires}. Use YYYY-MM-DD.")
    
    payload = {
        "company": company,
        "tier": tier,
        "expires": expires,
        "email": email,
        "issued": datetime.now().strftime("%Y-%m-%d")
    }
    
    payload_json = json.dumps(payload, separators=(',', ':'))
    payload_b64 = base64.b64encode(payload_json.encode('utf-8')).decode('utf-8')
    
    # Compute signature
    signing_secret = secret or os.environ.get("ADAPTIVE_K_SECRET", "VERTEX_ADAPTIVE_K_2026")
    sig_input = f"{payload_b64}{signing_secret}".encode('utf-8')
    signature = hashlib.sha256(sig_input).hexdigest()[:16]
    
    return f"{payload_b64}.{signature}"


def print_license_info(license_key: Optional[str] = None) -> None:
    """Print formatted license information.
    
    Args:
        license_key: License key to validate, or None for community tier
    """
    validator = LicenseValidator(license_key)
    info = validator.validate()
    
    print("\n" + "=" * 50)
    print("ADAPTIVE-K LICENSE INFORMATION")
    print("=" * 50)
    print(f"Tier:     {info.tier.upper()}")
    print(f"Status:   {'✓ Valid' if info.valid else '✗ Invalid'}")
    print(f"Company:  {info.company}")
    if info.expires:
        print(f"Expires:  {info.expires}")
    print(f"Message:  {info.message}")
    print("-" * 50)
    print("Features:")
    for feature in info.features:
        print(f"  • {feature}")
    print("-" * 50)
    print("Limits:")
    for limit, value in info.limits.items():
        print(f"  • {limit}: {value if value else 'Unlimited'}")
    print("=" * 50)
    print("\nUpgrade: https://adaptive-k.vertexdata.it/pricing")
    print("Support: amministrazione@vertexdata.it\n")


# CLI integration
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "validate":
            key = sys.argv[2] if len(sys.argv) > 2 else None
            print_license_info(key)
        elif sys.argv[1] == "generate":
            if len(sys.argv) < 5:
                print("Usage: python licensing.py generate <company> <tier> <expires>")
                sys.exit(1)
            key = generate_license_key(
                company=sys.argv[2],
                tier=sys.argv[3],
                expires=sys.argv[4]
            )
            print(f"Generated license key:\n{key}")
    else:
        print_license_info()
