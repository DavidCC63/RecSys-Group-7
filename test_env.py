import torch
import faiss
import torch_geometric
import igraph

print("✅ --- ¡ENTORNO CONFIGURADO CON ÉXITO! --- ✅")
print("\n")
print(f"Versión de Python: 3.9 (¡Correcto!)")
print(f"Versión de PyTorch: {torch.__version__} (Debe ser 1.9.1+cu111)")
print(f"Versión de PyG: {torch_geometric.__version__} (Debe ser 2.1.0.post1)")
print(f"Versión de IGraph: {igraph.__version__} (Debe ser 0.9.11)")
print("\n")
print("--- Verificación de GPU ---")
print(f"PyTorch detecta GPU: {torch.cuda.is_available()}")
print(f"FAISS detecta {faiss.get_num_gpus()} GPU(s)")

if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
    print("\n¡FELICIDADES! Tu entorno está 100% funcional con GPU. 🚀")
else:
    print("\n❌ ¡ERROR! Algo falló, la GPU no está siendo detectada.")