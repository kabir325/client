#!/usr/bin/env python3
"""
Smart AI Load Balancer Client v3.0
Enhanced fog computing client with intelligent model handling
"""

import grpc
from concurrent import futures
import threading
import time
import logging
import uuid
import socket
import subprocess
import argparse
from typing import Optional, List

# Import generated gRPC files
import load_balancer_pb2
import load_balancer_pb2_grpc

# Import performance evaluator
from performance_evaluator import PerformanceEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartLoadBalancerClient(load_balancer_pb2_grpc.LoadBalancerServicer):
    """Smart Load Balancer Client with enhanced capabilities"""
    
    def __init__(self, server_address: str):
        self.server_address = server_address
        self.client_id = self._generate_client_id()
        self.assigned_model = None
        self.model_info = None
        self.specs = PerformanceEvaluator.get_system_specs()
        self._running = False
        self.available_local_models = []
        
        # Progress tracking
        self.current_requests = {}  # request_id -> status info
        self._status_lock = threading.Lock()
        
        logger.info("🚀 Smart AI Load Balancer Client v3.0 Started")
        logger.info(f"Client ID: {self.client_id}")
        logger.info(f"Server: {self.server_address}")
        logger.info(f"Performance Score: {self.specs['performance_score']}")
        
        # Discover local models
        self._discover_local_models()
    
    def _generate_client_id(self) -> str:
        """Generate unique client ID"""
        hostname = socket.gethostname()
        return f"client-{hostname}-{str(uuid.uuid4())[:8]}"
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "127.0.0.1"
    
    def _discover_local_models(self) -> None:
        """Discover available local Ollama models"""
        try:
            logger.info("🔍 Discovering local models...")
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 1:
                            models.append(parts[0])
                
                self.available_local_models = models
                logger.info(f"📦 Found {len(models)} local models:")
                for model in models[:5]:  # Show first 5
                    logger.info(f"   • {model}")
                if len(models) > 5:
                    logger.info(f"   ... and {len(models) - 5} more")
            else:
                logger.warning("Could not list Ollama models")
                
        except Exception as e:
            logger.warning(f"Error discovering local models: {e}")
            self.available_local_models = []
    
    def register_with_server(self) -> bool:
        """Register this client with the smart server"""
        try:
            channel = grpc.insecure_channel(self.server_address)
            stub = load_balancer_pb2_grpc.LoadBalancerStub(channel)
            
            # Create registration request
            specs = load_balancer_pb2.SystemSpecs(
                cpu_cores=self.specs['cpu_cores'],
                cpu_frequency_ghz=self.specs['cpu_frequency_ghz'],
                ram_gb=int(self.specs['ram_gb']),
                gpu_info=self.specs['gpu_info'],
                gpu_memory_gb=self.specs['gpu_memory_gb'],
                os_info=self.specs['os_info'],
                performance_score=self.specs['performance_score']
            )
            
            request = load_balancer_pb2.ClientInfo(
                client_id=self.client_id,
                hostname=socket.gethostname(),
                ip_address=self._get_local_ip(),
                specs=specs
            )
            
            logger.info("📡 Registering with smart server...")
            response = stub.RegisterClient(request, timeout=15)
            
            if response.success:
                self.assigned_model = response.assigned_model
                self.model_info = response.model_info
                
                logger.info(f"✅ Smart registration successful!")
                logger.info(f"🤖 Assigned model: {self.assigned_model}")
                
                if self.model_info and self.model_info.parameters > 0:
                    params = self._format_parameters(self.model_info.parameters)
                    logger.info(f"📊 Model details: {params}, complexity {self.model_info.complexity_score}/10")
                
                logger.info(f"🌐 Total clients in network: {response.total_clients}")
                
                # Verify model availability
                self._verify_model_availability()
                
                channel.close()
                return True
            else:
                logger.error(f"❌ Registration failed: {response.message}")
                channel.close()
                return False
                
        except Exception as e:
            logger.error(f"❌ Error registering with server: {e}")
            return False
    
    def _verify_model_availability(self) -> None:
        """Verify that the assigned model is available locally"""
        if not self.assigned_model:
            return
        
        if self.assigned_model in self.available_local_models:
            logger.info(f"✅ Assigned model {self.assigned_model} is available locally")
        else:
            logger.warning(f"⚠️  Assigned model {self.assigned_model} not found locally")
            logger.info(f"💡 Attempting to pull model...")
            
            try:
                result = subprocess.run(['ollama', 'pull', self.assigned_model], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    logger.info(f"✅ Successfully pulled {self.assigned_model}")
                    self.available_local_models.append(self.assigned_model)
                else:
                    logger.error(f"❌ Failed to pull {self.assigned_model}: {result.stderr}")
            except Exception as e:
                logger.error(f"❌ Error pulling model: {e}")
    
    def ProcessAIRequest(self, request, context):
        """Process AI request from server with enhanced handling and progress tracking"""
        try:
            logger.info(f"📥 Received AI request: {request.request_id}")
            logger.info(f"🤖 Prompt: {request.prompt[:100]}{'...' if len(request.prompt) > 100 else ''}")
            logger.info(f"📊 Using model: {request.assigned_model}")
            
            # Initialize progress tracking
            with self._status_lock:
                self.current_requests[request.request_id] = {
                    'status': load_balancer_pb2.STARTING,
                    'progress': 0.0,
                    'step': 'Initializing request',
                    'start_time': time.time(),
                    'estimated_remaining': 0
                }
            
            start_time = time.time()
            
            # Process with enhanced Ollama handling
            response_text = self._process_with_enhanced_ollama_tracked(
                request.prompt, request.assigned_model, request.request_id
            )
            
            processing_time = time.time() - start_time
            
            # Mark as completed
            with self._status_lock:
                if request.request_id in self.current_requests:
                    self.current_requests[request.request_id]['status'] = load_balancer_pb2.COMPLETED
                    self.current_requests[request.request_id]['progress'] = 100.0
                    self.current_requests[request.request_id]['step'] = 'Completed'
            
            logger.info(f"✅ Processing completed in {processing_time:.1f}s")
            
            # Clean up tracking after a delay
            threading.Timer(30.0, self._cleanup_request, args=[request.request_id]).start()
            
            return load_balancer_pb2.AIResponse(
                request_id=request.request_id,
                success=True,
                response_text=response_text,
                processing_time=processing_time,
                client_id=self.client_id,
                model_used=request.assigned_model,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing AI request: {e}")
            
            # Mark as error
            with self._status_lock:
                if request.request_id in self.current_requests:
                    self.current_requests[request.request_id]['status'] = load_balancer_pb2.ERROR
                    self.current_requests[request.request_id]['step'] = f'Error: {str(e)}'
            
            return load_balancer_pb2.AIResponse(
                request_id=request.request_id,
                success=False,
                response_text=f"Error: {str(e)}",
                processing_time=0.0,
                client_id=self.client_id,
                model_used=request.assigned_model or "unknown",
                timestamp=int(time.time())
            )
    
    def GetProcessingStatus(self, request, context):
        """Get current processing status for progress tracking"""
        try:
            with self._status_lock:
                if request.request_id in self.current_requests:
                    req_info = self.current_requests[request.request_id]
                    
                    # Calculate estimated remaining time
                    elapsed = time.time() - req_info['start_time']
                    if req_info['progress'] > 0:
                        total_estimated = elapsed / (req_info['progress'] / 100.0)
                        remaining = max(0, total_estimated - elapsed)
                    else:
                        remaining = 0
                    
                    return load_balancer_pb2.StatusResponse(
                        request_id=request.request_id,
                        client_id=self.client_id,
                        status=req_info['status'],
                        progress_percentage=req_info['progress'],
                        current_step=req_info['step'],
                        estimated_remaining_seconds=int(remaining)
                    )
                else:
                    return load_balancer_pb2.StatusResponse(
                        request_id=request.request_id,
                        client_id=self.client_id,
                        status=load_balancer_pb2.IDLE,
                        progress_percentage=0.0,
                        current_step="No active request",
                        estimated_remaining_seconds=0
                    )
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return load_balancer_pb2.StatusResponse(
                request_id=request.request_id,
                client_id=self.client_id,
                status=load_balancer_pb2.ERROR,
                progress_percentage=0.0,
                current_step=f"Status error: {str(e)}",
                estimated_remaining_seconds=0
            )
    
    def _cleanup_request(self, request_id: str):
        """Clean up completed request tracking"""
        with self._status_lock:
            if request_id in self.current_requests:
                del self.current_requests[request_id]
    
    def _process_with_enhanced_ollama_tracked(self, prompt: str, model: str, request_id: str) -> str:
        """Process prompt with progress tracking"""
        try:
            # Update status: Starting processing
            with self._status_lock:
                if request_id in self.current_requests:
                    self.current_requests[request_id].update({
                        'status': load_balancer_pb2.PROCESSING,
                        'progress': 10.0,
                        'step': f'Starting Ollama model: {model}'
                    })
            
            logger.info(f"🔄 Processing with Ollama model: {model}")
            
            # Ensure model is available
            if model not in self.available_local_models:
                logger.warning(f"Model {model} not in local cache, attempting to use anyway...")
                with self._status_lock:
                    if request_id in self.current_requests:
                        self.current_requests[request_id].update({
                            'progress': 20.0,
                            'step': 'Model not cached, proceeding anyway'
                        })
            
            # Update status: Processing
            with self._status_lock:
                if request_id in self.current_requests:
                    self.current_requests[request_id].update({
                        'progress': 30.0,
                        'step': 'Running Ollama inference'
                    })
            
            # Run Ollama command (no timeout - let it run as long as needed)
            result = subprocess.run([
                'ollama', 'run', model, prompt
            ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            # Update status: Finalizing
            with self._status_lock:
                if request_id in self.current_requests:
                    self.current_requests[request_id].update({
                        'progress': 90.0,
                        'step': 'Processing Ollama response'
                    })
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if response:
                    logger.info("✅ Ollama processing successful")
                    return response
                else:
                    return "Empty response from Ollama"
            else:
                error_msg = result.stderr.strip() or "Unknown Ollama error"
                logger.error(f"Ollama error: {error_msg}")
                
                # Try fallback model if available
                fallback_response = self._try_fallback_model_tracked(prompt, model, request_id)
                if fallback_response:
                    return fallback_response
                
                return f"Ollama error: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error running Ollama: {e}")
            return f"Processing error: {str(e)}"
    
    def _try_fallback_model_tracked(self, prompt: str, failed_model: str, request_id: str) -> Optional[str]:
        """Try fallback model with progress tracking"""
        fallback_models = ['llama3.2:3b', 'llama3.2:1b', 'llama3:8b', 'llama2:7b']
        
        for fallback in fallback_models:
            if fallback != failed_model and fallback in self.available_local_models:
                try:
                    logger.info(f"🔄 Trying fallback model: {fallback}")
                    
                    with self._status_lock:
                        if request_id in self.current_requests:
                            self.current_requests[request_id].update({
                                'progress': 60.0,
                                'step': f'Trying fallback model: {fallback}'
                            })
                    
                    result = subprocess.run([
                        'ollama', 'run', fallback, prompt
                    ], capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    
                    if result.returncode == 0 and result.stdout.strip():
                        logger.info(f"✅ Fallback model {fallback} succeeded")
                        return f"[Processed with fallback model {fallback}]\n\n{result.stdout.strip()}"
                        
                except Exception as e:
                    logger.warning(f"Fallback model {fallback} also failed: {e}")
                    continue
        
        return None
    
    def _process_with_enhanced_ollama(self, prompt: str, model: str) -> str:
        """Process prompt using Ollama with enhanced error handling and fallbacks"""
        try:
            logger.info(f"🔄 Processing with Ollama model: {model}")
            
            # First, ensure model is available
            if model not in self.available_local_models:
                logger.warning(f"Model {model} not in local cache, attempting to use anyway...")
            
            # Run Ollama command with enhanced parameters
            result = subprocess.run([
                'ollama', 'run', model, prompt
            ], capture_output=True, text=True, timeout=90, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if response:
                    logger.info("✅ Ollama processing successful")
                    return response
                else:
                    return "Empty response from Ollama"
            else:
                error_msg = result.stderr.strip() or "Unknown Ollama error"
                logger.error(f"Ollama error: {error_msg}")
                
                # Try fallback model if available
                fallback_response = self._try_fallback_model(prompt, model)
                if fallback_response:
                    return fallback_response
                
                return f"Ollama error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            logger.error("Ollama processing timed out")
            return "Processing timed out (90s limit exceeded)"
        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama.")
            return "Ollama not installed on this system"
        except Exception as e:
            logger.error(f"Error running Ollama: {e}")
            return f"Processing error: {str(e)}"
    
    def _try_fallback_model(self, prompt: str, failed_model: str) -> Optional[str]:
        """Try to use a fallback model if the assigned model fails"""
        # Common fallback models in order of preference
        fallback_models = ['llama3.2:3b', 'llama3.2:1b', 'llama3:8b', 'llama2:7b']
        
        for fallback in fallback_models:
            if fallback != failed_model and fallback in self.available_local_models:
                try:
                    logger.info(f"🔄 Trying fallback model: {fallback}")
                    result = subprocess.run([
                        'ollama', 'run', fallback, prompt
                    ], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='ignore')
                    
                    if result.returncode == 0 and result.stdout.strip():
                        logger.info(f"✅ Fallback model {fallback} succeeded")
                        return f"[Processed with fallback model {fallback}]\n\n{result.stdout.strip()}"
                        
                except Exception as e:
                    logger.warning(f"Fallback model {fallback} also failed: {e}")
                    continue
        
        return None
    
    def HealthCheck(self, request, context):
        """Enhanced health check endpoint"""
        return load_balancer_pb2.HealthResponse(
            healthy=True,
            message=f"Client {self.client_id} healthy. Model: {self.assigned_model or 'None'}",
            connected_clients=1,
            active_models=1 if self.assigned_model else 0
        )
    
    def start_client_server(self):
        """Start the client's gRPC server to receive requests"""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        load_balancer_pb2_grpc.add_LoadBalancerServicer_to_server(self, server)
        
        listen_addr = '[::]:50052'
        server.add_insecure_port(listen_addr)
        server.start()
        
        logger.info(f"🌐 Smart client server listening on port 50052")
        logger.info("✅ Ready to receive AI requests from smart server")
        
        self._running = True
        
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down smart client...")
        finally:
            server.stop(0)
    
    def _format_parameters(self, parameters: int) -> str:
        """Format parameter count in human-readable form"""
        if parameters >= 1_000_000_000:
            return f"{parameters // 1_000_000_000}B parameters"
        elif parameters >= 1_000_000:
            return f"{parameters // 1_000_000}M parameters"
        else:
            return f"{parameters} parameters"
    
    def run(self):
        """Main client run method"""
        # Register with server
        if not self.register_with_server():
            logger.error("Failed to register with smart server. Exiting.")
            return
        
        # Start client server to receive requests
        self.start_client_server()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Smart AI Load Balancer Client v3.0')
    parser.add_argument('--server', default='localhost:50051',
                       help='Server address (default: localhost:50051)')
    
    args = parser.parse_args()
    
    client = SmartLoadBalancerClient(args.server)
    
    try:
        client.run()
    except KeyboardInterrupt:
        logger.info("Smart client stopped by user")
    except Exception as e:
        logger.error(f"Smart client error: {e}")

if __name__ == '__main__':
    main()