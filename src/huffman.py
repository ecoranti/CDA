from collections import Counter
from heapq import heappush, heappop

class HuffNode:
    def __init__(self, freq, sym=None, left=None, right=None):
        self.freq = freq
        self.sym = sym
        self.left = left
        self.right = right

    def __lt__(self, other):
        # Necesario para que heapq pueda comparar nodos
        return self.freq < other.freq

def build_code(symbols):
    """
    Construye el código de Huffman para una secuencia de símbolos (enteros).
    Devuelve:
      - code: dict {simbolo: 'cadena_de_bits'}
      - counts: Counter con frecuencias de cada símbolo
    """
    counts = Counter(symbols)
    heap = []

    # Inicializar heap con nodos hoja
    for s, f in counts.items():
        heappush(heap, HuffNode(f, sym=s))

    # Caso degenerado: solo hay un símbolo
    if len(heap) == 1:
        node = heappop(heap)
        return {node.sym: '0'}, counts

    # Construcción del árbol
    while len(heap) > 1:
        a = heappop(heap)
        b = heappop(heap)
        heappush(heap, HuffNode(a.freq + b.freq, left=a, right=b))

    root = heappop(heap)
    code = {}

    def walk(n, prefix):
        if n.sym is not None:
            code[n.sym] = prefix
            return
        walk(n.left, prefix + '0')
        walk(n.right, prefix + '1')

    walk(root, '')
    return code, counts

def encode(symbols):
    """
    Codifica 'symbols' (lista de enteros) con Huffman.
    Devuelve:
      - out_bits: lista de bits (0/1) de la codificación
      - code: diccionario de códigos {simbolo: 'bits'}
      - Lavg: longitud media de código (bits/símbolo)
    """
    code, counts = build_code(symbols)
    out_bits = []
    for s in symbols:
        out_bits.extend(int(c) for c in code[s])

    total = sum(counts.values())
    Lavg = sum(len(code[s]) * counts[s] for s in counts) / total
    return out_bits, code, Lavg
