"""
Setup validation tests for HyperPod Inference Operator.

These tests run before the main test suite to ensure the cluster
has the correct operator version installed.
"""
import subprocess
import requests


def get_public_operator_version():
    """Fetch the latest public operator version from GitHub."""
    url = "https://raw.githubusercontent.com/aws/sagemaker-hyperpod-cli/main/helm_chart/HyperPodHelmChart/charts/inference-operator/Chart.yaml"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    for line in response.text.split('\n'):
        if line.startswith('appVersion:'):
            version = line.split(':')[1].strip().strip('"')
            return version
    
    raise ValueError("Could not find appVersion in Chart.yaml")


def get_installed_operator_version():
    """Get the installed operator version from the cluster."""
    result = subprocess.run(
        ["helm", "list", "-A"],
        capture_output=True,
        text=True,
        check=True
    )
    
    for line in result.stdout.split('\n'):
        if 'hyperpod-inference-operator' in line:
            parts = line.split('\t')
            return parts[6].strip()
    
    raise ValueError("hyperpod-inference-operator not found in cluster")


def test_operator_version():
    """
    Verify cluster has the latest public operator version installed.
    
    This test ensures the cluster is running the correct operator version
    before running integration tests. Fails if versions don't match.
    """
    public_version = get_public_operator_version()
    installed_version = get_installed_operator_version()
    
    assert installed_version == public_version, (
        f"Operator version mismatch: "
        f"installed={installed_version}, public={public_version}. "
        f"Please upgrade the operator to the latest version."
    )
    
    print(f"\nOperator version validated: {installed_version}")
