#!/usr/bin/env python3
"""
Test script for Semantic Caching System
Demonstrates cache hits/misses with various query patterns
"""

import requests
import json
import time
from typing import List, Dict

API_URL = "http://localhost:3000"

def print_separator():
    print("\n" + "="*80 + "\n")

def send_query(query: str, force_refresh: bool = False) -> Dict:
    """Send a query to the API and return the response"""
    try:
        response = requests.post(
            f"{API_URL}/api/query",
            json={"query": query, "forceRefresh": force_refresh},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def get_stats() -> Dict:
    """Get cache statistics"""
    try:
        response = requests.get(f"{API_URL}/api/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def test_exact_duplicates():
    """Test 1: Exact duplicate queries"""
    print_separator()
    print("TEST 1: Exact Duplicate Queries")
    print_separator()
    
    query = "What is the capital of France?"
    
    print(f"Query 1: {query}")
    result1 = send_query(query)
    if result1:
        print(f"Source: {result1['metadata']['source']}")
        print(f"Latency: {result1['metadata']['latency_ms']}ms")
        print(f"Response: {result1['response'][:100]}...")
    
    time.sleep(1)
    
    print(f"\nQuery 2 (same): {query}")
    result2 = send_query(query)
    if result2:
        print(f"Source: {result2['metadata']['source']}")
        print(f"Latency: {result2['metadata']['latency_ms']}ms")
        if 'similarity_score' in result2['metadata']:
            print(f"Similarity: {result2['metadata']['similarity_score']}")

def test_semantic_similarity():
    """Test 2: Semantically similar queries with different wording"""
    print_separator()
    print("TEST 2: Semantically Similar Queries")
    print_separator()
    
    queries = [
        "Who was the first person to walk on the moon?",
        "Which astronaut first stepped on the lunar surface?",
        "Who made the first moon landing?",
        "Name the first human to set foot on the moon"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        result = send_query(query)
        if result:
            print(f"Source: {result['metadata']['source']}")
            print(f"Latency: {result['metadata']['latency_ms']}ms")
            if 'similarity_score' in result['metadata']:
                print(f"Similarity: {result['metadata']['similarity_score']}")
        time.sleep(0.5)

def test_unrelated_queries():
    """Test 3: Completely unrelated queries"""
    print_separator()
    print("TEST 3: Unrelated Queries (All Should Miss Cache)")
    print_separator()
    
    queries = [
        "What is quantum computing?",
        "How do you bake chocolate chip cookies?",
        "Who won the 2022 FIFA World Cup?",
        "Explain neural networks in simple terms"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        result = send_query(query)
        if result:
            print(f"Source: {result['metadata']['source']}")
            print(f"Latency: {result['metadata']['latency_ms']}ms")
        time.sleep(0.5)

def test_time_sensitive_queries():
    """Test 4: Time-sensitive vs evergreen queries"""
    print_separator()
    print("TEST 4: Time-Sensitive vs Evergreen Queries")
    print_separator()
    
    print("Time-sensitive queries (shorter cache TTL):")
    time_sensitive = [
        "What's the weather today in New York?",
        "What are the latest news headlines?",
        "Current stock price of Apple?"
    ]
    
    for query in time_sensitive:
        print(f"\n  Query: {query}")
        result = send_query(query)
        if result:
            print(f"  Source: {result['metadata']['source']}")
            print(f"  Response: {result['response'][:80]}...")
        time.sleep(0.5)
    
    print("\n\nEvergreen queries (longer cache TTL):")
    evergreen = [
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "When was the Declaration of Independence signed?"
    ]
    
    for query in evergreen:
        print(f"\n  Query: {query}")
        result = send_query(query)
        if result:
            print(f"  Source: {result['metadata']['source']}")
            print(f"  Response: {result['response'][:80]}...")
        time.sleep(0.5)

def test_force_refresh():
    """Test 5: Force refresh bypassing cache"""
    print_separator()
    print("TEST 5: Force Refresh")
    print_separator()
    
    query = "What is artificial intelligence?"
    
    print(f"Query 1 (normal): {query}")
    result1 = send_query(query)
    if result1:
        print(f"Source: {result1['metadata']['source']}")
    
    time.sleep(1)
    
    print(f"\nQuery 2 (force refresh): {query}")
    result2 = send_query(query, force_refresh=True)
    if result2:
        print(f"Source: {result2['metadata']['source']}")
        print(f"Note: Should be 'llm' even if cached")

def test_load_pattern():
    """Test 6: Rapid succession of queries (load testing)"""
    print_separator()
    print("TEST 6: Load Testing - Rapid Queries")
    print_separator()
    
    queries = [
        "What is machine learning?",
        "Explain machine learning",
        "Define machine learning",
        "What does machine learning mean?",
        "Machine learning basics"
    ]
    
    start_time = time.time()
    
    for i, query in enumerate(queries, 1):
        result = send_query(query)
        if result:
            print(f"{i}. {query[:50]}... -> {result['metadata']['source']} ({result['metadata']['latency_ms']}ms)")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average per query: {total_time/len(queries):.2f}s")

##removed for now - will add more comprehensive load testing in future iterations
def display_final_stats():
    """Display comprehensive statistics"""
    print_separator()
    print("FINAL CACHE STATISTICS")
    print_separator()
    
    stats = get_stats()
    if stats:
        print(json.dumps(stats, indent=2))
        
        print("\n\nKEY METRICS:")
        print(f"  • Cache Hit Rate: {stats['cache_hit_rate']}")
        print(f"  • Total Queries: {stats['total_queries']}")
        print(f"  • LLM Calls Saved: {stats['cache_hits']}")
        print(f"  • Average Latency: {stats['average_latency_ms']}ms")
        print(f"  • Cache Entries: {stats['total_cache_entries']}")

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SEMANTIC CACHING SYSTEM - TEST SUITE")
    print("="*80)
    
    # Wait for service to be ready
    print("\nWaiting for service to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                print("Service is ready!\n")
                break
        except:
            pass
        time.sleep(1)
        if i == max_retries - 1:
            print("Service did not become ready. Exiting.")
            return
    
    # Run tests
    try:
        test_exact_duplicates()
        test_semantic_similarity()
        test_unrelated_queries()
        test_time_sensitive_queries()
        test_force_refresh()
        test_load_pattern()
        
        print_separator()
        print("TEST SUITE COMPLETED!")
        print_separator()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError running tests: {e}")

if __name__ == "__main__":
    main()