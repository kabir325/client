#!/usr/bin/env python3
"""
Setup script for Tailscale-enabled load balancer client
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_tailscale():
    """Check if Tailscale is installed and running"""
    print("🔍 Checking Tailscale...")
    
    try:
        # Check if tailscale command exists
        result = subprocess.run(['tailscale', 'version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Tailscale installed: {result.stdout.strip()}")
        else:
            print("❌ Tailscale command failed")
            return False
    except FileNotFoundError:
        print("❌ Tailscale not found")
        print("💡 Install Tailscale from: https://tailscale.com/download")
        return False
    except Exception as e:
        print(f"❌ Error checking Tailscale: {e}")
        return False
    
    # Check Tailscale status
    try:
        result = subprocess.run(['tailscale', 'status'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Tailscale is running")
            return True
        else:
            print("❌ Tailscale not connected")
            print("💡 Run: tailscale up")
            return False
    except Exception as e:
        print(f"❌ Error checking Tailscale status: {e}")
        return False

def detect_tailscale_ip():
    """Detect and display Tailscale IP"""
    print("🌐 Detecting Tailscale IP...")
    
    try:
        result = subprocess.run(['tailscale', 'ip', '-4'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            tailscale_ip = result.stdout.strip()
            print(f"✅ Tailscale IP: {tailscale_ip}")
            return tailscale_ip
        else:
            print("❌ Could not get Tailscale IP")
            return None
    except Exception as e:
        print(f"❌ Error getting Tailscale IP: {e}")
        return None

def generate_grpc_files():
    """Generate gRPC files"""
    print("🔧 Generating gRPC files...")
    try:
        subprocess.run([sys.executable, 'generate_grpc_files.py'], check=True)
        print("✅ gRPC files generated")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate gRPC files: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 TAILSCALE LOAD BALANCER CLIENT SETUP")
    print("="*50)
    
    # Change to client directory
    if not os.path.exists('smart_load_balancer_client.py'):
        if os.path.exists('client/smart_load_balancer_client.py'):
            os.chdir('client')
        elif os.path.exists('v3/client/smart_load_balancer_client.py'):
            os.chdir('v3/client')
        else:
            print("❌ Could not find client directory")
            sys.exit(1)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Install dependencies
    if install_dependencies():
        success_count += 1
    
    # Step 2: Check Tailscale
    if check_tailscale():
        success_count += 1
    
    # Step 3: Detect Tailscale IP
    tailscale_ip = detect_tailscale_ip()
    if tailscale_ip:
        success_count += 1
    
    # Step 4: Generate gRPC files
    if generate_grpc_files():
        success_count += 1
    
    print("\n" + "="*50)
    print(f"📊 SETUP RESULTS: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("🎉 SETUP SUCCESSFUL!")
        print("\n💡 Next steps:")
        print("1. Get your server's Tailscale IP:")
        print("   tailscale ip -4")
        print("\n2. Start the client:")
        print(f"   python3 smart_load_balancer_client.py --server SERVER_TAILSCALE_IP:50051")
        print(f"\n3. Your client will use Tailscale IP: {tailscale_ip}")
        
    else:
        print("❌ SETUP INCOMPLETE")
        print("💡 Please fix the issues above and run setup again")

if __name__ == "__main__":
    main()