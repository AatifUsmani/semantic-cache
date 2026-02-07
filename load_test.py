#!/usr/bin/env python3
"""
Load Testing Script for Semantic Caching System
Tests performance under various load conditions
"""

import requests
import time
import concurrent.futures
import statistics
from typing import List, Dict
import random

API_URL = "http://localhost:3000"

# Sample queries for load testing
SAMPLE_QUERIES = [
    # Factual queries (should cache well)
    "What is the capital of France?",
    "Who invented the telephone?",
    "When was the first computer built?",
    "What is photosynthesis?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "When did World War 2 end?",
    "Who wrote Hamlet?",
    "What is the largest ocean?",
    "What is DNA?",
    
    # Variations (should hit cache with semantic matching)
    "What's the capital city of France?",
    "Who created the telephone?",
    "When was the first computing machine invented?",
    "Explain photosynthesis",
    "Who is the artist of Mona Lisa?",
    "How fast does light travel?",
    "When did the Second World War conclude?",
    "Who authored Hamlet?",
    "Which is the biggest ocean?",
    "What does DNA stand for?",
    
    # Time-sensitive queries
    "What's the weather today?",
    "Latest news headlines",
    "Current stock market prices",
    "Today's sports scores",
    "What time is it now?",
]

class LoadTestResults:
    def __init__(self):
        self.latencies: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.total_requests = 0
    
    def add_result(self, latency: float, source: str, success: bool):
        self.total_requests += 1
        if success:
            self.latencies.append(latency)
            if source == 'cache':
                self.cache_hits += 1
            else:
                self.cache_misses += 1
        else:
            self.errors += 1
    
    def get_summary(self) -> Dict:
        if not self.latencies:
            return {
                'error': 'No successful requests',
                'total_requests': self.total_requests,
                'errors': self.errors
            }
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': len(self.latencies),
            'errors': self.errors,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': f"{(self.cache_hits / len(self.latencies) * 100):.2f}%",
            'avg_latency_ms': f"{statistics.mean(self.latencies) * 1000:.2f}",
            'median_latency_ms': f"{statistics.median(self.latencies) * 1000:.2f}",
            'p95_latency_ms': f"{statistics.quantiles(self.latencies, n=20)[18] * 1000:.2f}",
            'p99_latency_ms': f"{statistics.quantiles(self.latencies, n=100)[98] * 1000:.2f}",
            'min_latency_ms': f"{min(self.latencies) * 1000:.2f}",
            'max_latency_ms': f"{max(self.latencies) * 1000:.2f}",
        }

def send_query(query: str) -> tuple:
    """Send a query and return (latency, source, success)"""
    start_time = time.time()
    try:
        response = requests.post(
            f"{API_URL}/api/query",
            json={"query": query},
            timeout=30
        )
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            source = data.get('metadata', {}).get('source', 'unknown')
            return (latency, source, True)
        else:
            return (latency, 'error', False)
    except Exception as e:
        latency = time.time() - start_time
        print(f"Error: {e}")
        return (latency, 'error', False)

def sequential_load_test(num_requests: int = 50) -> LoadTestResults:
    """Test with sequential requests"""
    print(f"\n{'='*60}")
    print(f"Sequential Load Test: {num_requests} requests")
    print(f"{'='*60}\n")
    
    results = LoadTestResults()
    start_time = time.time()
    
    for i in range(num_requests):
        query = random.choice(SAMPLE_QUERIES)
        latency, source, success = send_query(query)
        results.add_result(latency, source, success)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_requests} requests...")
    
    total_time = time.time() - start_time
    
    summary = results.get_summary()
    summary['total_time_seconds'] = f"{total_time:.2f}"
    summary['requests_per_second'] = f"{num_requests / total_time:.2f}"
    
    return summary

def concurrent_load_test(num_requests: int = 50, max_workers: int = 10) -> LoadTestResults:
    """Test with concurrent requests"""
    print(f"\n{'='*60}")
    print(f"Concurrent Load Test: {num_requests} requests, {max_workers} workers")
    print(f"{'='*60}\n")
    
    results = LoadTestResults()
    start_time = time.time()
    
    queries = [random.choice(SAMPLE_QUERIES) for _ in range(num_requests)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_query, query) for query in queries]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            latency, source, success = future.result()
            results.add_result(latency, source, success)
            completed += 1
            
            if completed % 10 == 0:
                print(f"Completed {completed}/{num_requests} requests...")
    
    total_time = time.time() - start_time
    
    summary = results.get_summary()
    summary['total_time_seconds'] = f"{total_time:.2f}"
    summary['requests_per_second'] = f"{num_requests / total_time:.2f}"
    
    return summary

def burst_load_test(bursts: int = 5, burst_size: int = 20, burst_delay: float = 2.0):
    """Test with burst traffic pattern"""
    print(f"\n{'='*60}")
    print(f"Burst Load Test: {bursts} bursts of {burst_size} requests")
    print(f"{'='*60}\n")
    
    all_results = LoadTestResults()
    burst_summaries = []
    
    for burst_num in range(bursts):
        print(f"\nBurst {burst_num + 1}/{bursts}...")
        
        burst_results = LoadTestResults()
        queries = [random.choice(SAMPLE_QUERIES) for _ in range(burst_size)]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as executor:
            futures = [executor.submit(send_query, query) for query in queries]
            
            for future in concurrent.futures.as_completed(futures):
                latency, source, success = future.result()
                burst_results.add_result(latency, source, success)
                all_results.add_result(latency, source, success)
        
        burst_time = time.time() - start_time
        summary = burst_results.get_summary()
        summary['burst_time_seconds'] = f"{burst_time:.2f}"
        burst_summaries.append(summary)
        
        print(f"Burst {burst_num + 1} completed in {burst_time:.2f}s")
        
        if burst_num < bursts - 1:
            time.sleep(burst_delay)
    
    return all_results.get_summary(), burst_summaries

def print_summary(title: str, summary: Dict):
    """Pretty print test summary"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")
    
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title():.<40} {value}")

def main():
    """Run all load tests"""
    print("\n" + "="*60)
    print("SEMANTIC CACHING SYSTEM - LOAD TESTING")
    print("="*60)
    
    # Wait for service
    print("\nWaiting for service to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                print("Service is ready!\n")
                break
        except:
            pass
        time.sleep(1)
    
    try:
        # Test 1: Sequential Load
        summary1 = sequential_load_test(num_requests=30)
        print_summary("Sequential Load Test Results", summary1)
        
        time.sleep(2)
        
        # Test 2: Concurrent Load
        summary2 = concurrent_load_test(num_requests=50, max_workers=10)
        print_summary("Concurrent Load Test Results", summary2)
        
        time.sleep(2)
        
        # Test 3: Burst Load
        summary3, burst_summaries = burst_load_test(bursts=3, burst_size=15)
        print_summary("Overall Burst Load Test Results", summary3)
        
        for i, burst_summary in enumerate(burst_summaries, 1):
            print(f"\nBurst {i} Details:")
            for key, value in burst_summary.items():
                print(f"  {key}: {value}")
        
        # Final Statistics
        print("\n" + "="*60)
        print("Getting Final Cache Statistics...")
        print("="*60 + "\n")
        
        response = requests.get(f"{API_URL}/api/stats")
        if response.status_code == 200:
            import json
            print(json.dumps(response.json(), indent=2))
        
        print("\n" + "="*60)
        print("LOAD TESTING COMPLETED!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nLoad testing interrupted by user")
    except Exception as e:
        print(f"\n\nError during load testing: {e}")

if __name__ == "__main__":
    main()