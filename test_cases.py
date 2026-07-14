"""
Recon Framework
Test Cases - API Endpoint Testing

Run this after starting the FastAPI server (port 8000).
"""

import requests
import json
import sys

# Force UTF-8 encoding for stdout and stderr to prevent Windows UnicodeEncodeErrors
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

API_BASE = "http://localhost:8000"

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0


def print_header(title):
    print(f"\n{'='*70}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{'='*70}")


def test_case(name, method, url, payload=None, expected_status=200, check_fields=None):
    global passed, failed
    print(f"\n{BOLD}[TEST] {name}{RESET}")
    print(f"  {method} {url}")
    
    try:
        if method == "GET":
            r = requests.get(url, timeout=10)
        elif method == "POST":
            r = requests.post(url, json=payload, timeout=10)
        
        status_ok = r.status_code == expected_status
        
        if status_ok:
            data = r.json()
            print(f"  Status: {GREEN}{r.status_code} OK{RESET}")
            
            # Check required fields
            fields_ok = True
            if check_fields:
                for field in check_fields:
                    if field not in data:
                        print(f"  {RED}MISSING field: {field}{RESET}")
                        fields_ok = False
            
            if fields_ok:
                print(f"  Result: {GREEN}PASSED{RESET}")
                passed += 1
            else:
                print(f"  Result: {RED}FAILED - Missing fields{RESET}")
                failed += 1
            
            # Print relevant response data
            if "prediction" in data:
                print(f"  {YELLOW}Prediction: {data['prediction']}{RESET}")
                print(f"  {YELLOW}Confidence: {data['confidence']*100:.1f}%{RESET}")
                print(f"  {YELLOW}Threat Level: {data['threat_level']}{RESET}")
                print(f"  {YELLOW}Probabilities: {data['all_probabilities']}{RESET}")
                print(f"  {YELLOW}Recommendation: {data['recommendation'][:80]}...{RESET}")
            elif "accuracy" in data:
                print(f"  {YELLOW}Model: {data.get('model_name')}{RESET}")
                print(f"  {YELLOW}Accuracy: {data['accuracy']*100:.2f}%{RESET}")
                print(f"  {YELLOW}Classes: {data.get('target_classes')}{RESET}")
            elif "status" in data:
                print(f"  {YELLOW}Status: {data['status']}{RESET}")
                print(f"  {YELLOW}Model Loaded: {data.get('model_loaded')}{RESET}")
            
            return data
        else:
            print(f"  Status: {RED}{r.status_code} (expected {expected_status}){RESET}")
            print(f"  Result: {RED}FAILED{RESET}")
            failed += 1
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"  {RED}ERROR: Cannot connect to API at {API_BASE}{RESET}")
        print(f"  {RED}Make sure FastAPI is running: uvicorn main:app --port 8000{RESET}")
        failed += 1
        return None
    except Exception as e:
        print(f"  {RED}ERROR: {str(e)}{RESET}")
        failed += 1
        return None


# ============================================================
# TEST CASES
# ============================================================

print_header("Recon Framework - API Test Suite")

# ----------------------------------------------------------
# TEST 1: Health Check
# ----------------------------------------------------------
print_header("Test 1: Health Check Endpoint")
test_case(
    name="GET / - Health Check",
    method="GET",
    url=f"{API_BASE}/",
    check_fields=["status", "message", "model_loaded"]
)

# ----------------------------------------------------------
# TEST 2: Model Info
# ----------------------------------------------------------
print_header("Test 2: Model Info Endpoint")
test_case(
    name="GET /api/model-info - Model Information",
    method="GET",
    url=f"{API_BASE}/api/model-info",
    check_fields=["model_name", "accuracy", "n_features", "target_classes", "feature_names"]
)

# ----------------------------------------------------------
# TEST 3: Feature Names
# ----------------------------------------------------------
print_header("Test 3: Feature Names Endpoint")
test_case(
    name="GET /api/features - Feature Names",
    method="GET",
    url=f"{API_BASE}/api/features",
    check_fields=["feature_names", "categorical_columns", "total_features"]
)

# ----------------------------------------------------------
# TEST 4: Normal Traffic (should predict 'normal')
# ----------------------------------------------------------
print_header("Test 4: Normal HTTP Traffic")
test_case(
    name="POST /api/predict - Normal HTTP Traffic",
    method="POST",
    url=f"{API_BASE}/api/predict",
    payload={
        "features": {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 267,
            "dst_bytes": 14515,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 4,
            "srv_count": 4,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 155,
            "dst_host_srv_count": 255,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.01,
            "dst_host_srv_diff_host_rate": 0.03,
            "dst_host_serror_rate": 0.01,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0
        }
    },
    check_fields=["prediction", "confidence", "threat_level", "all_probabilities", "recommendation"]
)

# ----------------------------------------------------------
# TEST 5: Anomaly - Port Scan / Probe Attack
# ----------------------------------------------------------
print_header("Test 5: Port Scan / Probe Attack (Anomaly)")
test_case(
    name="POST /api/predict - Port Scan (REJ flag, many connections)",
    method="POST",
    url=f"{API_BASE}/api/predict",
    payload={
        "features": {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "private",
            "flag": "REJ",
            "src_bytes": 0,
            "dst_bytes": 0,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 229,
            "srv_count": 10,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 1.0,
            "srv_rerror_rate": 1.0,
            "same_srv_rate": 0.04,
            "diff_srv_rate": 0.06,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 255,
            "dst_host_srv_count": 10,
            "dst_host_same_srv_rate": 0.04,
            "dst_host_diff_srv_rate": 0.06,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 1.0,
            "dst_host_srv_rerror_rate": 1.0
        }
    },
    check_fields=["prediction", "confidence", "threat_level"]
)

# ----------------------------------------------------------
# TEST 6: Anomaly - SYN Flood / DoS Attack
# ----------------------------------------------------------
print_header("Test 6: SYN Flood / DoS Attack (Anomaly)")
test_case(
    name="POST /api/predict - SYN Flood (S0 flag, high count)",
    method="POST",
    url=f"{API_BASE}/api/predict",
    payload={
        "features": {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "telnet",
            "flag": "S0",
            "src_bytes": 0,
            "dst_bytes": 0,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 120,
            "srv_count": 120,
            "serror_rate": 1.0,
            "srv_serror_rate": 1.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 235,
            "dst_host_srv_count": 171,
            "dst_host_same_srv_rate": 0.73,
            "dst_host_diff_srv_rate": 0.07,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.69,
            "dst_host_srv_serror_rate": 0.95,
            "dst_host_rerror_rate": 0.02,
            "dst_host_srv_rerror_rate": 0.0
        }
    },
    check_fields=["prediction", "confidence", "threat_level"]
)

# ----------------------------------------------------------
# TEST 7: Normal SMTP Email Traffic
# ----------------------------------------------------------
print_header("Test 7: Normal SMTP Email Traffic")
test_case(
    name="POST /api/predict - Normal SMTP Traffic",
    method="POST",
    url=f"{API_BASE}/api/predict",
    payload={
        "features": {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "smtp",
            "flag": "SF",
            "src_bytes": 1022,
            "dst_bytes": 387,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 1,
            "srv_count": 3,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 1.0,
            "dst_host_count": 255,
            "dst_host_srv_count": 28,
            "dst_host_same_srv_rate": 0.11,
            "dst_host_diff_srv_rate": 0.72,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.72,
            "dst_host_srv_rerror_rate": 0.04
        }
    },
    check_fields=["prediction", "confidence", "threat_level"]
)

# ----------------------------------------------------------
# TEST 8: ICMP Smurf / Ping Flood Attack
# ----------------------------------------------------------
print_header("Test 8: ICMP Smurf Attack (Anomaly)")
test_case(
    name="POST /api/predict - ICMP Flood (high count, ecr_i service)",
    method="POST",
    url=f"{API_BASE}/api/predict",
    payload={
        "features": {
            "duration": 0,
            "protocol_type": "icmp",
            "service": "ecr_i",
            "flag": "SF",
            "src_bytes": 520,
            "dst_bytes": 0,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 511,
            "srv_count": 511,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 46,
            "dst_host_srv_count": 59,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 1.0,
            "dst_host_srv_diff_host_rate": 0.14,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0
        }
    },
    check_fields=["prediction", "confidence", "threat_level"]
)

# ----------------------------------------------------------
# TEST 9: Suspicious Telnet Session (Anomaly)
# ----------------------------------------------------------
print_header("Test 9: Suspicious Telnet Session (Anomaly)")
test_case(
    name="POST /api/predict - Suspicious Telnet (logged_in=0, SF)",
    method="POST",
    url=f"{API_BASE}/api/predict",
    payload={
        "features": {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "telnet",
            "flag": "SF",
            "src_bytes": 129,
            "dst_bytes": 174,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 1,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 1,
            "srv_count": 1,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 255,
            "dst_host_srv_count": 255,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.0,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.01,
            "dst_host_srv_serror_rate": 0.01,
            "dst_host_rerror_rate": 0.02,
            "dst_host_srv_rerror_rate": 0.02
        }
    },
    check_fields=["prediction", "confidence", "threat_level"]
)

# ----------------------------------------------------------
# TEST 10: Normal FTP Data Transfer
# ----------------------------------------------------------
print_header("Test 10: Normal FTP Data Transfer")
test_case(
    name="POST /api/predict - Normal FTP Data Transfer",
    method="POST",
    url=f"{API_BASE}/api/predict",
    payload={
        "features": {
            "duration": 2,
            "protocol_type": "tcp",
            "service": "ftp_data",
            "flag": "SF",
            "src_bytes": 12983,
            "dst_bytes": 0,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 1,
            "srv_count": 1,
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 134,
            "dst_host_srv_count": 86,
            "dst_host_same_srv_rate": 0.61,
            "dst_host_diff_srv_rate": 0.04,
            "dst_host_same_src_port_rate": 0.61,
            "dst_host_srv_diff_host_rate": 0.02,
            "dst_host_serror_rate": 0.0,
            "dst_host_srv_serror_rate": 0.0,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0
        }
    },
    check_fields=["prediction", "confidence", "threat_level"]
)

# ----------------------------------------------------------
# TEST 11: Flask Frontend Health
# ----------------------------------------------------------
print_header("Test 11: Flask Frontend Accessibility")
try:
    r = requests.get("http://localhost:5000/", timeout=10)
    if r.status_code == 200 and "Recon" in r.text:
        print(f"  {GREEN}PASSED{RESET} - Flask dashboard is accessible")
        print(f"  {YELLOW}Page size: {len(r.text)} bytes{RESET}")
        passed += 1
    else:
        print(f"  {RED}FAILED{RESET} - Unexpected response")
        failed += 1
except Exception as e:
    print(f"  {RED}FAILED{RESET} - {e}")
    failed += 1

# ----------------------------------------------------------
# TEST 12: Flask Predict Page
# ----------------------------------------------------------
print_header("Test 12: Flask Prediction Form Page")
try:
    r = requests.get("http://localhost:5000/predict", timeout=10)
    if r.status_code == 200 and "Analyze" in r.text:
        print(f"  {GREEN}PASSED{RESET} - Prediction form is accessible")
        passed += 1
    else:
        print(f"  {RED}FAILED{RESET} - Unexpected response")
        failed += 1
except Exception as e:
    print(f"  {RED}FAILED{RESET} - {e}")
    failed += 1

# ============================================================
# SUMMARY
# ============================================================
print(f"\n\n{'='*70}")
print(f"{BOLD}  TEST RESULTS SUMMARY{RESET}")
print(f"{'='*70}")
total = passed + failed
print(f"\n  Total Tests:  {total}")
print(f"  {GREEN}Passed:       {passed}{RESET}")
print(f"  {RED}Failed:       {failed}{RESET}")
print(f"  Success Rate: {passed/total*100:.1f}%")
print(f"\n{'='*70}")

if failed == 0:
    print(f"\n  {GREEN}{BOLD}ALL TESTS PASSED!{RESET}")
else:
    print(f"\n  {RED}{BOLD}{failed} TEST(S) FAILED{RESET}")

print(f"\n  Dashboard: http://localhost:5000")
print(f"  API Docs:  http://localhost:8000/docs")
print(f"{'='*70}\n")

sys.exit(0 if failed == 0 else 1)
