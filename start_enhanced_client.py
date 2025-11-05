#!/usr/bin/env python3
"""
Start Enhanced Smart AI Load Balancer Client v3.1
With progress tracking and no timeout constraints
"""

import subprocess
import sys
import os
import argparse

def main():
    """Start the enhanced smart client"""
    parser = argparse.ArgumentParser(description='Start Enhanced Smart AI Load Balancer Client v3.1')
    parser.add_argument('--server', default='localhost:50051',
                       help='Server address (default: localhost:50051)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Enhanced Smart AI Load Balancer Client v3.1")
    print("="*70)
    print(f"Server: {args.server}")
    print("🆕 NEW FEATURES:")
    print("   ✅ No timeout constraints - can process complex queries")
    print("   ✅ Real-time progress reporting to server")
    print("   ✅ Enhanced error handling and fallback support")
    print("   ✅ Automatic model pulling if missing")
    print("="*70)
    
    # Check if gRPC files exist
    if not os.path.exists('load_balancer_pb2.py'):
        print("📦 Generating enhanced gRPC files...")
        try:
            subprocess.run([sys.executable, 'generate_grpc_files.py'], check=True)
            os.chdir('../..')
        except subprocess.CalledProcessError:
            print("❌ Failed to generate gRPC files")
            print("Make sure grpcio-tools is installed: pip install grpcio-tools")
            return
    
    # Start the enhanced client
    print("📱 Starting enhanced smart load balancer client...")
    print("💡 Features:")
    print("   • Enhanced model handling")
    print("   • Auto-discovery")
    print("   • Fallback support")
    print("   • Progress tracking")
    print("   • No processing limits")
    print()
    
    try:
        subprocess.run([sys.executable, 'smart_load_balancer_client.py', '--server', args.server], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Enhanced smart client stopped by user")
    except Exception as e:
        print(f"❌ Enhanced smart client error: {e}")
    finally:
        os.chdir('../..')

if __name__ == '__main__':
    main()