from certifier.signer import KernelSigner


def test_sign_and_verify_roundtrip() -> None:
    signer = KernelSigner(b"test-secret")
    kernel = {"version": "1.0.0", "tom_parameters": {"homo_offset": 0.1}}
    signed = signer.sign(kernel)
    assert "signature" in signed
    assert len(signed["signature"]) == 64
    assert signer.verify(signed)


def test_verify_fails_with_wrong_secret() -> None:
    signer1 = KernelSigner(b"secret-1")
    signer2 = KernelSigner(b"secret-2")
    kernel = {"version": "1.0.0"}
    signed = signer1.sign(kernel)
    assert not signer2.verify(signed)


def test_verify_fails_tampered_payload() -> None:
    signer = KernelSigner(b"test-secret")
    kernel = {"version": "1.0.0", "tom_parameters": {}}
    signed = signer.sign(kernel)
    signed["tom_parameters"]["homo_offset"] = 99.9
    assert not signer.verify(signed)


def test_verify_no_signature() -> None:
    signer = KernelSigner(b"test-secret")
    assert not signer.verify({"version": "1.0.0"})
