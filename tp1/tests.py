import numpy as np
import scipy.linalg
from template_funciones import construye_adyacencia, calculaLU, calcula_matriz_C, calcula_pagerank

# TESTS
def run_all_tests():
    print("🔍 Empezando tests...\n")
    try:
        # === Test 1 ===
        print("Test 1: construye_adyacencia con matriz chica")
        D = np.array([
            [0, 1, 4, 2],
            [1, 0, 3, 8],
            [4, 3, 0, 8],
            [2, 8, 8, 0]
        ])
        expected_A = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ])
        A = construye_adyacencia(D, 2)
        assert np.array_equal(A, expected_A), f"❌ construye_adyacencia falló. Obtuvo:\n{A}"
        print("  ✅ construye_adyacencia pasó.")

        # === Test 2 ===
        print("\nTest 2: calculaLU con matriz 2x2")
        M = np.array([[4, 3],
                      [6, 3]], dtype=float)
        L_expected = np.array([[1, 0],
                               [1.5, 1]])
        U_expected = np.array([[4, 3],
                               [0, -1.5]])
        L, U = calculaLU(M)
        assert np.allclose(L, L_expected), f"❌ L incorrecta:\n{L}"
        assert np.allclose(U, U_expected), f"❌ U incorrecta:\n{U}"
        print("  ✅ calculaLU pasó.")

        # === Test 3 ===
        print("\nTest 3: calcula_matriz_C")
        A_C = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        expected_C = np.array([
            [0, 0.5, 0],
            [1, 0, 1],
            [0, 0.5, 0]
        ])
        C = calcula_matriz_C(A_C)
        assert np.allclose(C, expected_C), f"❌ calcula_matriz_C falló. Obtuvo:\n{C}"
        print("  ✅ calcula_matriz_C pasó.")

        # === Test 4 ===
        print("\nTest 4: calcula_pagerank con grafo simétrico")
        A_PR = np.array([
            [0, 1],
            [1, 0]
        ])
        p = calcula_pagerank(A_PR, 0.15)
        assert p.shape == (1, 2), f"❌ Vector PageRank con forma incorrecta: {p.shape}"
        assert np.allclose(p[0, 0], p[0, 1], atol=1e-6), f"❌ PageRank desigual: {p}"
        print("  ✅ calcula_pagerank pasó en grafo simétrico.")

        # === Test 5 ===
        print("\nTest 5: construye_adyacencia con nodo aislado")
        D = np.array([
            [0, 1, 1e9],
            [1, 0, 1e9],
            [1e9, 1e9, 0]
        ])
        A = construye_adyacencia(D, 1)
        assert np.sum(A[2]) == 0, f"❌ Nodo aislado no detectado. Obtuvo fila 2:\n{A[2]}"
        print("  ✅ construye_adyacencia pasó con nodo aislado.")

        # === Test 6 ===
        print("\nTest 6: calcula_matriz_C con nodo sin conexiones")
        A = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0]
        ])
        C = calcula_matriz_C(A)
        assert np.allclose(C[2], [0, 0, 0]), f"❌ Nodo sin conexiones mal procesado. Obtuvo fila 2:\n{C[2]}"
        print("  ✅ calcula_matriz_C pasó con nodo sin conexiones.")

        print("\n🎉 TODOS LOS TESTS PASARON CORRECTAMENTE.\n")

    except AssertionError as e:
        print(str(e))
        print("\n🚨 Test fallido. Revisá el mensaje de error.\n")