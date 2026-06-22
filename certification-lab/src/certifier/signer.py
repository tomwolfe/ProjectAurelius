import hashlib
import hmac
import json
from typing import Any


class KernelSigner:
    """Cryptographic signer for Aurelius Certified Kernels.

    Generates the ``signature`` field of ``aurelius_kernel.json`` using
    HMAC-SHA256 over a canonical JSON serialisation of the kernel fields
    (excluding the signature itself), combined with a secret salt.

    Parameters
    ----------
    secret : bytes
        Secret key for HMAC signing. In production this should come from a
        secure vault or environment variable.

    Examples
    --------
    >>> signer = KernelSigner(b"my-secret-key")
    >>> kernel = {"version": "1.0.0", "tom_parameters": {}}
    >>> signed = signer.sign(kernel)
    >>> signer.verify(signed)
    True
    """

    def __init__(self, secret: bytes) -> None:
        self.secret = secret

    def sign(self, kernel: dict[str, Any]) -> dict[str, Any]:
        """Sign a kernel dict by adding a ``signature`` field.

        Parameters
        ----------
        kernel : dict
            Kernel dictionary (must NOT already have a ``signature`` field
            populated, or the signature will be computed over the old value).

        Returns
        -------
        dict
            The kernel dict with the ``signature`` field set.
        """
        payload = {k: v for k, v in kernel.items() if k != "signature"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        kernel["signature"] = hmac.new(
            self.secret, canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return kernel

    def verify(self, kernel: dict[str, Any]) -> bool:
        """Verify the signature on a signed kernel dict.

        Parameters
        ----------
        kernel : dict
            A signed kernel dictionary containing a ``signature`` field.

        Returns
        -------
        bool
            True if the signature is valid, False otherwise.
        """
        stored = kernel.get("signature", "")
        if not stored:
            return False
        payload = {k: v for k, v in kernel.items() if k != "signature"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hmac.new(
            self.secret, canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, stored)
