#!/usr/bin/env python3
"""
Test script for AudioSep and MDX-Net separation.
Runs both separators on phone_pay_dial.wav and reports results.
"""

import requests
import time
import json
import sys

API_BASE = "http://127.0.0.1:8000"
FILE_ID = "bd552a3b"  # phone_pay_dial.wav

def poll_job(job_id, timeout=300):
    """Poll job until completion or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{API_BASE}/api/status/{job_id}")
            status = res.json()

            if status["status"] == "completed":
                return {"success": True, "result": status.get("result"), "message": status.get("message")}
            elif status["status"] == "error":
                return {"success": False, "error": status.get("message")}

            print(f"  Progress: {status.get('progress', 0)*100:.0f}% - {status.get('message', '')}")
            time.sleep(2)
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {"success": False, "error": "Timeout"}

def test_mdxnet():
    """Test MDX-Net separation."""
    print("\n" + "="*60)
    print("TEST: MDX-Net Separation")
    print("="*60)

    try:
        res = requests.post(f"{API_BASE}/api/separate/{FILE_ID}?separator=mdxnet")
        data = res.json()

        if "job_id" not in data:
            return {"success": False, "error": f"No job_id returned: {data}"}

        print(f"Job started: {data['job_id']}")
        result = poll_job(data["job_id"])

        if result["success"]:
            print(f"\nMDX-Net SUCCESS!")
            print(f"  Stems: {list(result['result']['stems'].keys())}")
            print(f"  Processing time: {result['result']['processing_time']:.2f}s")
        else:
            print(f"\nMDX-Net FAILED: {result['error']}")

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}

def test_audiosep(prompt="telephone"):
    """Test AudioSep separation with a prompt."""
    print("\n" + "="*60)
    print(f"TEST: AudioSep Separation (prompt: '{prompt}')")
    print("="*60)

    try:
        res = requests.post(f"{API_BASE}/api/separate/{FILE_ID}?separator=audiosep&prompt={prompt}")
        data = res.json()

        if "job_id" not in data:
            return {"success": False, "error": f"No job_id returned: {data}"}

        print(f"Job started: {data['job_id']}")
        result = poll_job(data["job_id"], timeout=600)  # AudioSep can be slow

        if result["success"]:
            print(f"\nAudioSep SUCCESS!")
            print(f"  Stems: {list(result['result']['stems'].keys())}")
            print(f"  Processing time: {result['result']['processing_time']:.2f}s")
        else:
            print(f"\nAudioSep FAILED: {result['error']}")

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}

def run_tests():
    """Run all separation tests."""
    print("\n" + "#"*60)
    print("# SEPARATION TEST SUITE")
    print(f"# File: phone_pay_dial.wav (ID: {FILE_ID})")
    print("#"*60)

    # Check server health
    try:
        health = requests.get(f"{API_BASE}/api/health").json()
        print(f"\nServer status: {health['status']}")
    except Exception as e:
        print(f"\nERROR: Server not reachable - {e}")
        return {"mdxnet": None, "audiosep": None}

    results = {}

    # Test MDX-Net
    results["mdxnet"] = test_mdxnet()

    # Test AudioSep
    results["audiosep"] = test_audiosep("telephone")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, result in results.items():
        if result and result.get("success"):
            print(f"  {name}: PASSED")
        else:
            print(f"  {name}: FAILED - {result.get('error', 'Unknown error') if result else 'Not run'}")
            all_passed = False

    print("="*60)

    return results, all_passed

if __name__ == "__main__":
    results, all_passed = run_tests()
    sys.exit(0 if all_passed else 1)
