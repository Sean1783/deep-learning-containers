"""
Container Contract Tests for vLLM Images
Tests that vLLM images meet the operator's container requirements
"""
import subprocess
import time
import json
import os


def test_deploy_vllm_pod():
    """
    Test 1: Deploy vLLM and verify pod starts
    Validates: YAML valid, operator creates pod, image pulls successfully
    """
    endpoint_name = "baseline"
    namespace = "default"
    
    # Get path to baseline.yaml
    manifest_path = os.path.join(
        os.path.dirname(__file__),
        "manifests",
        "baseline.yaml"
    )
    
    print(f"Deploying from {manifest_path}")
    
    # Apply baseline.yaml
    result = subprocess.run(
        ["kubectl", "apply", "-f", manifest_path],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"Applied InferenceEndpointConfig: {result.stdout}")
    
    # Wait for pod to reach Running state
    pod_name = wait_for_pod_running(endpoint_name, namespace, timeout=120)
    
    print(f"Test passed: Pod {pod_name} is Running")


def test_health_endpoint():
    """
    Test 2: Verify health endpoint responds
    Validates: Container exposes /health endpoint on port 8000
    """
    endpoint_name = "baseline"
    namespace = "default"
    
    # Wait for pod to be fully ready (all containers)
    pod_name = wait_for_pod_ready(endpoint_name, namespace, timeout=300)
    print(f"Testing health endpoint on pod: {pod_name}")
    
    # Curl health endpoint from inside pod
    result = subprocess.run(
        [
            "kubectl", "exec", pod_name, "-n", namespace,
            "--", "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
            "http://localhost:8000/health"
        ],
        capture_output=True,
        text=True,
        check=True
    )
    
    status_code = result.stdout.strip()
    print(f"Health endpoint returned: {status_code}")
    
    assert status_code == "200", f"Expected 200, got {status_code}"
    print("Test passed: Health endpoint responds with 200")


def wait_for_pod_ready(endpoint_name, namespace, timeout=300):
    """
    Wait for pod to be fully ready (all containers ready)
    Returns pod name when ready
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        result = subprocess.run(
            [
                "kubectl", "get", "pods",
                "-n", namespace,
                "-l", f"app={endpoint_name}",
                "-o", "json"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        pods = json.loads(result.stdout)
        
        if not pods["items"]:
            print(f"No pods found yet for {endpoint_name}, waiting...")
            time.sleep(5)
            continue
        
        pod = pods["items"][0]
        pod_name = pod["metadata"]["name"]
        
        # Check if all containers are ready
        container_statuses = pod["status"].get("containerStatuses", [])
        ready_count = sum(1 for c in container_statuses if c.get("ready", False))
        total_count = len(container_statuses)
        
        print(f"Pod {pod_name}: {ready_count}/{total_count} containers ready")
        
        if ready_count == total_count and total_count > 0:
            print(f"Pod {pod_name} is fully ready")
            return pod_name
        
        time.sleep(10)
    
    raise TimeoutError(f"Pod {endpoint_name} did not become ready within {timeout} seconds")


def wait_for_pod_running(endpoint_name, namespace, timeout=120):
    """
    Wait for pod with label app=<endpoint_name> to reach Running state
    Returns pod name when Running
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Get pods with label selector
        result = subprocess.run(
            [
                "kubectl", "get", "pods",
                "-n", namespace,
                "-l", f"app={endpoint_name}",
                "-o", "json"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        pods = json.loads(result.stdout)
        
        if not pods["items"]:
            print(f"No pods found yet for {endpoint_name}, waiting...")
            time.sleep(5)
            continue
        
        pod = pods["items"][0]
        pod_name = pod["metadata"]["name"]
        phase = pod["status"]["phase"]
        
        print(f"Pod {pod_name} status: {phase}")
        
        if phase == "Running":
            return pod_name
        
        if phase == "Failed":
            raise RuntimeError(f"Pod {pod_name} failed to start")
        
        time.sleep(5)
    
    raise TimeoutError(f"Pod for {endpoint_name} did not reach Running state within {timeout} seconds")


def get_pod_name(endpoint_name, namespace):
    """
    Get pod name for endpoint
    """
    result = subprocess.run(
        [
            "kubectl", "get", "pods",
            "-n", namespace,
            "-l", f"app={endpoint_name}",
            "-o", "jsonpath={.items[0].metadata.name}"
        ],
        capture_output=True,
        text=True,
        check=True
    )
    
    pod_name = result.stdout.strip()
    if not pod_name:
        raise RuntimeError(f"No pod found for {endpoint_name}")
    
    return pod_name
