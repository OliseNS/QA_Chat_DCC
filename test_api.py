#!/usr/bin/env python3
"""
Simple script to test the Healthcare AI Assistant API endpoints.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000/api"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat():
    """Test the chat endpoint."""
    print("\nTesting chat endpoint...")
    try:
        payload = {
            "message": "What are the different types of dialysis treatments?",
            "history": []
        }
        
        response = requests.post(
            f"{BASE_URL}/chat",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result['response'][:200]}...")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_search():
    """Test the search endpoint."""
    print("\nTesting search endpoint...")
    try:
        payload = {
            "query": "home hemodialysis benefits",
            "top_k": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Found {len(result['results'])} results")
            if result['results']:
                print(f"First result similarity: {result['results'][0].get('similarity', 'N/A')}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_calculate():
    """Test the calculate endpoint."""
    print("\nTesting calculate endpoint...")
    try:
        payload = {
            "type": "bmi",
            "parameters": {
                "weight_kg": 70,
                "height_m": 1.75
            }
        }
        
        response = requests.post(
            f"{BASE_URL}/calculate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"BMI Result: {result['result']}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_stats():
    """Test the stats endpoint."""
    print("\nTesting stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Stats: {result['stats']}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("Healthcare AI Assistant API Test Script")
    print("=" * 40)
    
    # Wait a moment for the server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health),
        ("Chat", test_chat),
        ("Search", test_search),
        ("Calculate", test_calculate),
        ("Stats", test_stats)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'-' * 20}")
        try:
            success = test_func()
            results.append((name, success))
            print(f"{name}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"{name}: ERROR - {e}")
            results.append((name, False))
    
    print(f"\n{'=' * 40}")
    print("Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, success in results:
        print(f"  {name}: {'✓' if success else '✗'}")
    
    if passed == total:
        print("\nAll tests passed! The API is working correctly.")
    else:
        print(f"\n{total - passed} test(s) failed. Please check the API.")

if __name__ == "__main__":
    main()