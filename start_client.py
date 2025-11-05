#!/usr/bin/env python3
"""
Start Smart AI Load Balancer Client v3.0
"""

import subprocess
import sys
import os
import argparse

def main():
    """Start the smart client with proper setup"""
    parser = argparse.ArgumentParser(description='Start Smart AI Load Balancer Client v3.0')
    parser.add_argument('--server', default='localhost:50051',
                       help='Server address (default: localhost:50051)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Smart AI Load Balancer Client v3.0")
    print("="*60)
    print(f"Server: {args.server}")
    
    # Check if gRPC files exist
    if not os.path.exists('load_balancer_pb2.py'):
        print("📦 Generating gRPC files...")
        try:
            subprocess.run([sys.executable, 'generate_grpc_files.py'], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to generate gRPC files")
            print("Make sure grpcio-tools is installed: pip install grpcio-tools")
            return
    
    # Start the smart client
    print("📱 Starting smart load balancer client...")
    print("💡 Features: Enhanced model handling, auto-discovery, fallback support")
    try:
        subprocess.run([sys.executable, 'smart_load_balancer_client.py', '--server', args.server], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Smart client stopped by user")
    except Exception as e:
        print(f"❌ Smart client error: {e}")

if __name__ == '__main__':
    main()