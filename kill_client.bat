@echo off
echo Killing any running client processes...
echo.

REM Kill Python processes running the client
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :50052') do (
    echo Found process using port 50052: %%a
    taskkill /F /PID %%a
)

echo.
echo Done! Port 50052 should now be free.
echo You can now start the client again.
pause
