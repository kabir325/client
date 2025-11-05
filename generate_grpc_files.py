#!/usr/bin/env python3
"""
Generate gRPC files from proto definition
"""

import subprocess
import sys
import os

def generate_grpc_files():
    """Generate gRPC Python files from proto definition"""
    try:
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            "--proto_path=.",
            "--python_out=.",
            "--grpc_python_out=.",
            "load_balancer.proto"
        ]
        
        print("Generating gRPC files...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ gRPC files generated successfully!")
        print("Generated files:")
        print("  - load_balancer_pb2.py")
        print("  - load_balancer_pb2_grpc.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating gRPC files: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_grpc_files()