import argparse
import threading
import sys
import os

# filepath: /home/ignatus/Documentos/Github/captura-de-movimiento/app.py

# Ajusta PYTHONPATH si es necesario
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "container"))
sys.path.insert(0, os.path.join(ROOT, "signhandler"))

# --- imports de container ---
import container.app as container_app   # container/app.py define main(), CameraProcessor, etc.

# --- imports de signhandler ---
from signhandler.app import app as signhandler_app  # Importa directamente la instancia de Flask

def run_server(host="0.0.0.0", port=5000):
    """Arranca el servidor REST para firma y comparación."""
    print(f"🔑 Iniciando SIGNHANDLER server en {host}:{port}")
    signhandler_app.run(host=host, port=port)

def run_camera_flow(ip_camera, dest_ip, dest_port, interval=5):
    """Ejecuta sólo la captura+envío de imágenes."""
    print(f"📷 Iniciando CAMERA flow contra {dest_ip}:{dest_port}")
    proc = container_app.CameraProcessor(
        carpeta_capturas="capturas",
        ip_destino=dest_ip,
        puerto=dest_port
    )
    proc.ejecutar_flujo_completo(ip_camera)

def run_full(ip_camera, dest_ip, dest_port, host="0.0.0.0", port=5000):
    """Ejecuta both: server REST + captura/envío."""
    t = threading.Thread(target=run_server, args=(host, port), daemon=True)
    t.start()
    run_camera_flow(ip_camera, dest_ip, dest_port)

def main():
    p = argparse.ArgumentParser(description="Un solo ejecutable: signhandler + container")
    sub = p.add_subparsers(dest="cmd", required=True)

    # servidor
    srv = sub.add_parser("server", help="Arranca API REST de firma/comparación")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=5000)

    # cámara
    cam = sub.add_parser("camera", help="Captura y envía imágenes")
    cam.add_argument("ip_camera", help="IP de la cámara (o 0 para webcam)")
    cam.add_argument("--dest-ip", default="192.168.1.100")
    cam.add_argument("--dest-port", type=int, default=8080)

    # todo junto
    full = sub.add_parser("full", help="Arranca server y flujo de cámara en paralelo")
    full.add_argument("ip_camera", help="IP de la cámara (o 0 para webcam)")
    full.add_argument("--dest-ip", default="192.168.1.100")
    full.add_argument("--dest-port", type=int, default=8080)
    full.add_argument("--host", default="0.0.0.0")
    full.add_argument("--port", type=int, default=5000)

    args = p.parse_args()

    if args.cmd == "server":
        run_server(args.host, args.port)
    elif args.cmd == "camera":
        run_camera_flow(args.ip_camera, args.dest_ip, args.dest_port)
    elif args.cmd == "full":
        run_full(args.ip_camera, args.dest_ip, args.dest_port, args.host, args.port)

if __name__ == "__main__":
    main()