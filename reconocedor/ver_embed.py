import pickle

RUTA_EMBEDDINGS = "reconocedor/embeddings.pkl"

try:
    with open(RUTA_EMBEDDINGS, "rb") as f:
        data = pickle.load(f)

    print(f"\n[🔍] Nombres registrados en {RUTA_EMBEDDINGS}:\n")
    for nombre in data.keys():
        print(f" - {nombre}")
    print(f"\n[✔] Total: {len(data)} persona(s) registrada(s)")

except FileNotFoundError:
    print(f"[❌] No se encontró el archivo: {RUTA_EMBEDDINGS}")
except EOFError:
    print(f"[⚠️] El archivo está vacío o corrupto.")
except Exception as e:
    print(f"[‼️] Error al cargar el archivo: {e}")