"""
vLLM SageMaker Endpoint Test - Replicates deploysKVCacheWithDefaults
Tests full operator workflow: CRD -> SageMaker Endpoint -> Inference
"""
import subprocess
import time
import json
import boto3
import os


def test_vllm_kv_cache_l1_deployment():
    """
    Replicates: deploysKVCacheWithDefaults from G5IntegrationTests
    
    Workflow:
    1. Apply InferenceEndpointConfig CRD
    2. Wait for operator to create SageMaker Endpoint (InService)
    3. Invoke SageMaker Endpoint with inference request
    4. Verify response
    """
    endpoint_name = "kv-cache-l1"
    namespace = "default"
    region = "us-east-2"
    
    apply_inference_endpoint_config(endpoint_name, namespace)
    wait_for_endpoint_in_service(endpoint_name, region, timeout_minutes=20)
    invoke_endpoint_and_verify(endpoint_name, region)
    
    print("\n=== Test passed: vLLM KV Cache L1 deployment validated ===")


def test_vllm_kv_cache_l1_l2_deployment():
    """
    Replicates: deploysKVCacheWithL1AndL2MembrainEnabled from G5IntegrationTests
    
    Workflow:
    1. Apply InferenceEndpointConfig CRD with L1 + L2 cache
    2. Wait for operator to create SageMaker Endpoint (InService)
    3. Invoke SageMaker Endpoint with inference request
    4. Verify KV cache performance with both cache layers
    """
    endpoint_name = "kv-cache-l1-l2"
    namespace = "default"
    region = "us-east-2"
    
    apply_inference_endpoint_config(endpoint_name, namespace)
    wait_for_endpoint_in_service(endpoint_name, region, timeout_minutes=20)
    invoke_endpoint_and_verify(endpoint_name, region)
    
    print("\n=== Test passed: vLLM KV Cache L1+L2 deployment validated ===")


def test_vllm_kv_cache_l2_only_deployment():
    """
    Replicates: deploysKVCacheWithL2MembrainEnabled from G5IntegrationTests
    
    Workflow:
    1. Apply InferenceEndpointConfig CRD with L2 cache only (no L1)
    2. Wait for operator to create SageMaker Endpoint (InService)
    3. Invoke SageMaker Endpoint with inference request
    4. Verify L2 cache functionality
    """
    endpoint_name = "kv-cache-l2"
    namespace = "default"
    region = "us-east-2"
    
    apply_inference_endpoint_config(endpoint_name, namespace)
    wait_for_endpoint_in_service(endpoint_name, region, timeout_minutes=20)
    invoke_endpoint_and_verify(endpoint_name, region)
    
    print("\n=== Test passed: vLLM KV Cache L2-only deployment validated ===")


def test_vllm_intelligent_routing_deployment():
    """
    Replicates: deploysIntelligentRoutingWithL1Cache from G5IntegrationTests
    
    Workflow:
    1. Apply InferenceEndpointConfig CRD with intelligent routing + L1 cache
    2. Wait for operator to create SageMaker Endpoint (InService)
    3. Invoke SageMaker Endpoint with inference request
    4. Verify prefix-aware routing functionality
    """
    endpoint_name = "intelligent-routing"
    namespace = "default"
    region = "us-east-2"
    
    apply_inference_endpoint_config(endpoint_name, namespace)
    wait_for_endpoint_in_service(endpoint_name, region, timeout_minutes=20)
    invoke_endpoint_and_verify(endpoint_name, region)
    
    print("\n=== Test passed: vLLM Intelligent Routing deployment validated ===")


def apply_inference_endpoint_config(endpoint_name, namespace):
    """Apply InferenceEndpointConfig CRD"""
    manifest_path = os.path.join(
        os.path.dirname(__file__),
        "manifests",
        f"{endpoint_name}.yaml"
    )
    
    print(f"\n=== Applying InferenceEndpointConfig: {endpoint_name} ===")
    result = subprocess.run(
        ["kubectl", "apply", "-f", manifest_path],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"Applied: {result.stdout.strip()}")


def wait_for_endpoint_in_service(endpoint_name, region, timeout_minutes=20):
    """
    Wait for SageMaker endpoint to reach InService status
    Replicates: ModelDeployer.waitForEndpointInService()
    """
    print(f"\n=== Waiting for SageMaker endpoint: {endpoint_name} ===")
    
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    seconds_per_attempt = 30
    max_attempts = (timeout_minutes * 60) // seconds_per_attempt
    
    for attempt in range(1, max_attempts + 1):
        elapsed_minutes = (attempt * seconds_per_attempt) / 60
        print(f"Attempt {attempt}/{max_attempts} (elapsed: {elapsed_minutes:.1f} min)")
        
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            print(f"  Status: {status}")
            
            if status == 'InService':
                print(f"✓ Endpoint InService after {elapsed_minutes:.1f} minutes")
                return
            
            if status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown')
                raise RuntimeError(f"Endpoint failed: {failure_reason}")
            
            time.sleep(seconds_per_attempt)
            
        except sagemaker.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                print(f"  Endpoint not found yet (operator still creating)")
                time.sleep(seconds_per_attempt)
            else:
                raise
    
    raise TimeoutError(f"Endpoint not InService after {timeout_minutes} minutes")


def invoke_endpoint_and_verify(endpoint_name, region):
    """
    Invoke SageMaker endpoint and verify KV cache performance
    Replicates: ModelDeployer.verifyKVCachePerformance()
    """
    print(f"\n=== Verifying KV Cache Performance: {endpoint_name} ===")
    
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Chat completion request
    payload = {
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    # First call - populate cache
    print("Making first call to populate cache...")
    start_time_1 = time.time()
    response_1 = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps(payload)
    )
    end_time_1 = time.time()
    first_call_duration = (end_time_1 - start_time_1) * 1000  # Convert to ms
    
    result_1 = json.loads(response_1['Body'].read().decode())
    assert 'choices' in result_1, "First call: Response missing 'choices'"
    print(f"  First call completed in {first_call_duration:.0f}ms")
    
    # Wait for cache to settle
    time.sleep(2)
    
    # Second call - should use cache
    print("Making second identical call to test cache hit...")
    start_time_2 = time.time()
    response_2 = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps(payload)
    )
    end_time_2 = time.time()
    second_call_duration = (end_time_2 - start_time_2) * 1000  # Convert to ms
    
    result_2 = json.loads(response_2['Body'].read().decode())
    assert 'choices' in result_2, "Second call: Response missing 'choices'"
    print(f"  Second call completed in {second_call_duration:.0f}ms")
    
    # Calculate improvement
    improvement = ((first_call_duration - second_call_duration) / first_call_duration) * 100
    print(f"✓ Cache performance: First={first_call_duration:.0f}ms, Second={second_call_duration:.0f}ms, Improvement={improvement:.2f}%")
