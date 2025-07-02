import cv2
import requests
import numpy as np
from urllib.parse import urlparse
import socket
from concurrent.futures import ThreadPoolExecutor

def capture_from_ip_camera(ip_address, port, username=None, password=None):
    """
    Capture images from an IP camera
    
    Args:
        ip_address (str): IP address of the camera
        port (int): Port of the camera
        username (str): Username for authentication (optional)
        password (str): Password for authentication (optional)
    """
    if not port:
        port = 8081
    # Common IP camera stream URLs
    stream_urls = [
        f"http://{ip_address}:{port}/video/mjpg.cgi",
        f"http://{ip_address}:{port}/mjpg/video.mjpg",
        f"http://{ip_address}:{port}/videostream.cgi",
        f"http://{ip_address}:{port}/video",
        f"rtsp://{ip_address}:{port}/stream1"
    ]
    
    # Try different stream URLs
    for url in stream_urls:
        try:
            print(f"Trying URL: {url}")
            
            # Add authentication if provided
            if username and password:
                url = url.replace("://", f"://{username}:{password}@")
            
            # Open video capture
            cap = cv2.VideoCapture(url)
            
            if cap.isOpened():
                print(f"Successfully connected to {url}")
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    
                    if ret:
                        # Display the frame
                        cv2.imshow('IP Camera Feed', frame)
                        
                        # Save frame (optional)
                        if cv2.waitKey(1) & 0xFF == ord('s'):
                            filename = f"capture_{frame_count}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"Saved {filename}")
                            frame_count += 1
                        
                        # Exit on 'q' key
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print("Failed to read frame")
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                return True
                
        except Exception as e:
            print(f"Error with {url}: {e}")
            continue
    
    print("Could not connect to any camera stream")
    return False

def scan_network_for_cameras(network_base="192.168.1"):
    """
    Scan network for potential IP cameras
    
    Args:
        network_base (str): Network base (e.g., "192.168.1")
    """
    
    def check_camera_port(ip):
        common_ports = [80, 8080, 554, 8554]
        for port in common_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((ip, port))
                sock.close()
                if result == 0:
                    return f"{ip}:{port}"
            except:
                pass
        return None
    
    # Generate IP range
    ip_range = [f"{network_base}.{i}" for i in range(1, 255)]
    
    print("Scanning network for cameras...")
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(check_camera_port, ip_range))
    
    cameras = [result for result in results if result]
    return cameras

if __name__ == "__main__":
    # Example usage
    print("IP Camera Capture Tool")
    print("1. Scan network for cameras")
    print("2. Connect to specific IP")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        cameras = scan_network_for_cameras()
        if cameras:
            print(f"Found potential cameras: {cameras}")
            for camera in cameras:
                ip, port = camera.split(':')
                username = input(f"Username for {ip}:{port} (optional): ") or None
                password = input(f"Password for {ip}:{port} (optional): ") or None
                capture_from_ip_camera(ip, port, username, password)
        else:
            print("No cameras found")
    
    elif choice == "2":
        ip = input("Enter camera IP address: ")
        port = input("Enter camera port: ")
        username = input("Username (optional): ") or None
        password = input("Password (optional): ") or None
        capture_from_ip_camera(ip, port, username, password)