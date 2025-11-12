@echo off
echo ========================================
echo Smart Load Balancer Client Starter
echo ========================================
echo.

set /p SERVER_IP="Enter server IP address (default: localhost): "
if "%SERVER_IP%"=="" set SERVER_IP=localhost

echo.
echo Starting client connecting to %SERVER_IP%:50051...
echo.

python smart_load_balancer_client.py --server %SERVER_IP%:50051

pause
