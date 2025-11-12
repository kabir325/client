# Client - Smart AI Load Balancer

The client component runs on fog computing nodes and processes AI queries using Ollama.

## 📦 Components

- **smart_load_balancer_client.py** - Main client implementation
- **performance_evaluator.py** - Hardware performance evaluation

## 🚀 Setup

### 1. Install Ollama

**Windows:**
- Download from https://ollama.ai
- Run installer
- Verify: `ollama --version`

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull AI Models
```bash
# Small model (fast, less accurate)
ollama pull llama3.2:1b

# Medium model (balanced)
ollama pull llama3.2:3b

# Large model (slow, more accurate)
ollama pull llama3.1:8b
```

### 3. Verify Models
```bash
ollama list
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 5. Generate gRPC Files
```bash
python generate_grpc_files.py
```

## ▶️ Starting the Client

### Local Testing (Same Machine as Server)
```bash
python smart_load_balancer_client.py --server localhost:50051
```

### Remote Server
```bash
python smart_load_balancer_client.py --server SERVER_IP:50051
```

Replace `SERVER_IP` with your server's IP address.

### Windows Batch File
```bash
start_client.bat
```

This will prompt for the server IP address.

## 🔧 Configuration

### Ports
- **50052** - Client gRPC server (receives requests from main server)

### Environment
- Python 3.8+
- Ollama installed and running
- At least 8GB RAM (16GB+ recommended)
- GPU optional but recommended

### Network
- Must be able to reach server on port 50051
- Server must be able to reach client on port 50052
- Firewall must allow both ports

## 🎯 How It Works

### 1. Startup Process
```
Start → Evaluate hardware → Discover models → Connect to server → Register → Receive model assignment → Ready
```

### 2. Registration
The client automatically:
- Detects CPU cores, frequency, RAM, GPU
- Calculates performance score (0-100)
- Discovers available Ollama models
- Sends specs to server
- Receives optimal model assignment

### 3. Query Processing
```
Receive query → Process with assigned model → Send response → Report progress
```

### 4. Performance Scoring

**Formula:** CPU (40%) + RAM (30%) + GPU (30%)

**Examples:**
- High-end (16 cores, 32GB, RTX 4090): Score ~95
- Mid-range (8 cores, 16GB, RTX 3060): Score ~85
- Low-end (4 cores, 8GB, Intel GPU): Score ~55

## 📊 Expected Output

### Successful Registration
```
🚀 Smart AI Load Balancer Client v3.0 Started
Client ID: client-HOSTNAME-abc123
Server: 192.168.1.100:50051
Performance Score: 85.0
📡 Registering with smart server...
✅ Smart registration successful!
🤖 Assigned model: llama3.2:3b
📊 Model details: 3B, complexity 5/10
🌐 Total clients in network: 2
✅ Assigned model llama3.2:3b is available locally
🌐 Smart client server listening on port 50052
✅ Ready to receive AI requests from smart server
```

### Processing Query
```
📥 Received AI request: abc-123-def
🤖 Prompt: What is artificial intelligence?
📊 Using model: llama3.2:3b
🔄 Processing with Ollama model: llama3.2:3b
✅ Ollama processing successful
✅ Processing completed in 12.3s
```

## 🐛 Troubleshooting

### Ollama Not Found
```bash
# Check if installed
ollama --version

# If not, install from https://ollama.ai
```

### No Models Available
```bash
# List models
ollama list

# Pull a model
ollama pull llama3.2:3b
```

### Can't Connect to Server
**Check server IP:**
```bash
ping SERVER_IP
```

**Check port connectivity:**
```bash
telnet SERVER_IP 50051
```

**Common issues:**
- Wrong IP address
- Firewall blocking port 50051
- Server not running
- Different network/VPN

### Model Not Found Error
The client will automatically try fallback models:
1. Assigned model
2. llama3.2:3b
3. llama3.2:1b
4. llama3:8b
5. llama2:7b

If all fail, pull at least one model:
```bash
ollama pull llama3.2:3b
```

### Port 50052 Already in Use

**Quick Fix (Windows):**
```bash
# Run the kill script
kill_client.bat
```

**Manual Fix:**
```bash
# Windows - Find and kill process
netstat -ano | findstr :50052
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:50052 | xargs kill -9
```

**Common Cause:** Previous client instance still running. Always close properly with Ctrl+C.

### Server Can't Reach Client
- Check firewall allows port 50052
- Verify client IP is reachable from server
- Ensure both on same network/VPN
- For Tailscale: Client will auto-detect Tailscale IP

## 🌐 Network Configuration

### Local Network
Both server and client on same LAN:
- Use local IP (e.g., 192.168.1.100)
- Ensure no firewall blocking

### Remote Network
Server and client on different networks:
- Use public IP or VPN (Tailscale recommended)
- Configure port forwarding if needed
- Ensure firewall rules allow traffic

### Tailscale VPN (Recommended for Remote)
1. Install Tailscale on both machines
2. Client auto-detects Tailscale IP
3. Use Tailscale IP for server address
4. No port forwarding needed

## 📝 Files

```
client/
├── smart_load_balancer_client.py    # Main client
├── performance_evaluator.py         # Performance scoring
├── load_balancer.proto              # gRPC protocol
├── generate_grpc_files.py           # Proto compiler
├── requirements.txt                 # Dependencies
├── setup.bat                        # Setup script
└── start_client.bat                 # Starter script
```

## 🎓 Advanced Usage

### Multiple Clients
Run multiple clients on different machines:
```bash
# Machine 1
python smart_load_balancer_client.py --server SERVER_IP:50051

# Machine 2
python smart_load_balancer_client.py --server SERVER_IP:50051

# Machine 3
python smart_load_balancer_client.py --server SERVER_IP:50051
```

Each will get a model based on its performance.

### Custom Model Assignment
The server assigns models automatically, but you can:
1. Pull specific models you want to use
2. Server will only assign from available models
3. Better hardware = more complex models

### Monitoring
Watch the client terminal for:
- Registration status
- Model assignment
- Query processing
- Performance metrics
- Errors and warnings

### Logging
Adjust logging level in the code:
```python
logging.basicConfig(level=logging.INFO)  # or DEBUG, WARNING, ERROR
```

## 🔐 Security Notes

**Development Mode:**
- No authentication
- Unencrypted gRPC

**For Production:**
- Enable gRPC TLS/SSL
- Add client authentication
- Restrict network access
- Use VPN for remote clients

## 💡 Tips

1. **GPU Acceleration**: If you have a GPU, Ollama will use it automatically
2. **Model Size**: Larger models need more RAM (1B ≈ 2GB, 3B ≈ 6GB, 8B ≈ 16GB)
3. **Performance**: Close unnecessary applications for better performance
4. **Network**: Use wired connection for stability
5. **Multiple Models**: Pull multiple models for flexibility
