@echo off
echo ========================================
echo Smart Load Balancer Client Setup
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Generating gRPC files...
python generate_grpc_files.py

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To start the client:
echo   1. Run: python smart_load_balancer_client.py --server SERVER_IP:50051
echo   2. Or run: start_client.bat
echo.
echo Make sure Ollama is installed and running!
echo   - Install: https://ollama.ai
echo   - Pull models: ollama pull llama3.2:3b
echo.
pause
