#!/usr/bin/env python3
"""
Cross-platform Performance Evaluator v3.0
Enhanced version with better detection and scoring
"""

import psutil
import platform
import subprocess
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    """Cross-platform system performance evaluator"""
    
    @staticmethod
    def get_system_specs() -> Dict[str, Any]:
        """Get comprehensive system specifications"""
        try:
            specs = {
                'cpu_cores': psutil.cpu_count(logical=True),
                'cpu_frequency_ghz': PerformanceEvaluator._get_cpu_frequency(),
                'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'gpu_info': PerformanceEvaluator._get_gpu_info(),
                'gpu_memory_gb': PerformanceEvaluator._get_gpu_memory(),
                'os_info': PerformanceEvaluator._get_os_info(),
                'performance_score': 0.0
            }
            
            # Calculate performance score
            specs['performance_score'] = PerformanceEvaluator._calculate_performance_score(specs)
            
            return specs
            
        except Exception as e:
            logger.error(f"Error getting system specs: {e}")
            return PerformanceEvaluator._get_fallback_specs()
    
    @staticmethod
    def _get_cpu_frequency() -> float:
        """Get CPU frequency in GHz"""
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current:
                return round(freq.current / 1000, 2)
            else:
                if platform.system() == "Linux":
                    return PerformanceEvaluator._get_linux_cpu_freq()
                return 2.5
        except:
            return 2.5
    
    @staticmethod
    def _get_linux_cpu_freq() -> float:
        """Get CPU frequency from /proc/cpuinfo on Linux"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'cpu MHz' in line:
                        freq_mhz = float(line.split(':')[1].strip())
                        return round(freq_mhz / 1000, 2)
        except:
            pass
        return 2.5
    
    @staticmethod
    def _get_gpu_info() -> str:
        """Get GPU information across platforms"""
        system = platform.system()
        
        try:
            if system == "Windows":
                return PerformanceEvaluator._get_windows_gpu()
            elif system == "Darwin":
                return PerformanceEvaluator._get_macos_gpu()
            elif system == "Linux":
                return PerformanceEvaluator._get_linux_gpu()
            else:
                return "Unknown GPU"
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}")
            return "No GPU detected"
    
    @staticmethod
    def _get_windows_gpu() -> str:
        """Get GPU info on Windows using wmic"""
        try:
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = [line.strip() for line in lines[1:] if line.strip()]
                return gpus[0] if gpus else "No GPU detected"
        except:
            pass
        return "Windows GPU (detection failed)"
    
    @staticmethod
    def _get_macos_gpu() -> str:
        """Get GPU info on macOS using system_profiler"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Chipset Model:' in line:
                        return line.split(':')[1].strip()
        except:
            pass
        return "macOS GPU (detection failed)"
    
    @staticmethod
    def _get_linux_gpu() -> str:
        """Get GPU info on Linux using lspci"""
        try:
            result = subprocess.run(
                ['lspci', '-nn'], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA compatible controller' in line or '3D controller' in line:
                        parts = line.split(': ')
                        if len(parts) > 1:
                            gpu_name = parts[1].split(' [')[0]
                            return gpu_name
        except:
            pass
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        
        return "Linux GPU (detection failed)"
    
    @staticmethod
    def _get_gpu_memory() -> float:
        """Get GPU memory in GB"""
        system = platform.system()
        
        try:
            if system == "Linux":
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    memory_mb = int(result.stdout.strip().split('\n')[0])
                    return round(memory_mb / 1024, 2)
            
            return 4.0
            
        except:
            return 0.0
    
    @staticmethod
    def _get_os_info() -> str:
        """Get OS information"""
        try:
            system = platform.system()
            release = platform.release()
            
            if system == "Windows":
                return f"Windows {release}"
            elif system == "Darwin":
                return f"macOS {release}"
            elif system == "Linux":
                try:
                    with open('/etc/os-release', 'r') as f:
                        for line in f:
                            if line.startswith('PRETTY_NAME='):
                                return line.split('=')[1].strip().strip('"')
                except:
                    pass
                return f"Linux {release}"
            else:
                return f"{system} {release}"
        except:
            return "Unknown OS"
    
    @staticmethod
    def _calculate_performance_score(specs: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100) with enhanced algorithm"""
        try:
            # Enhanced CPU scoring (0-40 points)
            cpu_base = min(20, specs['cpu_cores'] * 1.5)
            cpu_freq = min(20, specs['cpu_frequency_ghz'] * 6)
            cpu_score = cpu_base + cpu_freq
            
            # Enhanced RAM scoring (0-30 points)
            ram_score = min(30, specs['ram_gb'] * 1.5)
            
            # Enhanced GPU scoring (0-30 points)
            gpu_score = 0
            gpu_info = specs['gpu_info'].lower()
            
            # More detailed GPU scoring
            if 'nvidia' in gpu_info:
                if any(x in gpu_info for x in ['rtx 40', 'a100', 'h100']):
                    gpu_score = 30
                elif any(x in gpu_info for x in ['rtx 30', 'v100', 'a40']):
                    gpu_score = 28
                elif any(x in gpu_info for x in ['rtx 20', 'gtx 16', 'quadro']):
                    gpu_score = 25
                elif 'rtx' in gpu_info:
                    gpu_score = 22
                elif 'gtx' in gpu_info:
                    gpu_score = 18
                else:
                    gpu_score = 15
            elif 'amd' in gpu_info or 'radeon' in gpu_info:
                if any(x in gpu_info for x in ['rx 7', 'rx 6']):
                    gpu_score = 25
                elif any(x in gpu_info for x in ['rx 5', 'vega']):
                    gpu_score = 20
                else:
                    gpu_score = 15
            elif 'intel' in gpu_info:
                if 'arc' in gpu_info:
                    gpu_score = 20
                elif 'iris xe' in gpu_info or 'iris' in gpu_info:
                    gpu_score = 12
                else:
                    gpu_score = 8
            elif any(x in gpu_info for x in ['apple', 'm1', 'm2', 'm3']):
                if 'm3' in gpu_info:
                    gpu_score = 28
                elif 'm2' in gpu_info:
                    gpu_score = 25
                elif 'm1' in gpu_info:
                    gpu_score = 22
                else:
                    gpu_score = 20
            else:
                gpu_score = 5
            
            total_score = cpu_score + ram_score + gpu_score
            return round(min(100, total_score), 1)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0
    
    @staticmethod
    def _get_fallback_specs() -> Dict[str, Any]:
        """Fallback specs if detection fails"""
        return {
            'cpu_cores': 4,
            'cpu_frequency_ghz': 2.5,
            'ram_gb': 8.0,
            'gpu_info': 'Unknown GPU',
            'gpu_memory_gb': 0.0,
            'os_info': platform.system(),
            'performance_score': 50.0
        }

if __name__ == "__main__":
    evaluator = PerformanceEvaluator()
    specs = evaluator.get_system_specs()
    
    print("System Specifications:")
    print(f"CPU Cores: {specs['cpu_cores']}")
    print(f"CPU Frequency: {specs['cpu_frequency_ghz']} GHz")
    print(f"RAM: {specs['ram_gb']} GB")
    print(f"GPU: {specs['gpu_info']}")
    print(f"GPU Memory: {specs['gpu_memory_gb']} GB")
    print(f"OS: {specs['os_info']}")
    print(f"Performance Score: {specs['performance_score']}")