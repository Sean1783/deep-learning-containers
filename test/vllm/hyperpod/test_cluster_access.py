import subprocess

def test_cluster_connection():
    result = subprocess.run(
        ["kubectl", "get", "nodes"],
        capture_output=True,
        text=True,
        check=True
    )
    assert "Ready" in result.stdout

def test_list_inference_endpoints():
    result = subprocess.run(
        ["kubectl", "get", "inferenceendpointconfig", "-A"],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"\n{result.stdout}")
    assert result.returncode == 0

def test_list_pods():
    result = subprocess.run(
        ["kubectl", "get", "pods", "-A"],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"\n{result.stdout}")
    assert result.returncode == 0
