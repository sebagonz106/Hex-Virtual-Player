# Noobex, Jugador Virtual de Hex: Implementación y Análisis Experimental de Monte Carlo Tree Search

## Descripción

Este proyecto implementa un jugador virtual competitivo para el juego Hex mediante estrategias progresivamente mejoradas de Monte Carlo Tree Search (MCTS). Se exploran técnicas contemporáneas incluyendo evaluación rápida de acciones (RAVE), reciclaje de árbol de búsqueda, paralelización, y análisis de tableros reducidos bajo restricción temporal estricta de 5 segundos por movimiento.

Hex es un juego de mesa estratégico combinatorio de información perfecta para dos jugadores, jugado en un tablero con celdas hexagonales. El objetivo es conectar dos bordes opuestos del tablero con piezas del propio color. A pesar de sus reglas simples, el alto factor de ramificación del árbol de juego lo convierte en un banco de pruebas interesante para algoritmos de búsqueda en inteligencia artificial.

## Estructura del Proyecto

```
Hex-Virtual-Player/
├── main.py                          # Punto de entrada principal
├── board.py                         # Implementación base del tablero Hex
├── players/
│   ├── player.py                    # Interfaz abstracta de jugador
│   ├── random_player.py             # Jugador baseline aleatorio
│   ├── basic_MCTS_player.py         # MCTS convencional (referencia)
│   ├── recycler_MCTS_player.py      # MCTS con reciclaje de árbol
│   ├── RAVE_MCTS_player.py          # MCTS con evaluación rápida de acciones
│   ├── progressive_MCTS_player.py   # MCTS con parámetros adaptativos
│   ├── parallelized_MCTS_player.py  # MCTS paralelizado con sincronización
│   ├── final_player.py              # SmartPlayer (solución final)
│   └── utils/
│       ├── board_optimized.py              # Representación Union-Find optimizada
│       ├── board_optimized_with_fillin.py  # Union-Find con análisis de celdas inferiores
│       ├── expandability_manager.py        # Gestión de expansión de nodos
│       ├── game_phase.py                   # Clasificación de fase de juego
│       ├── early_check.py                  # Heurísticas de detección temprana
│       └── __init__.py
├── Sebastian_Gonzalez_Alfonso/
│   └── solution.py                  # Jugador final entregable (SmartPlayer)
|   └── estrategia.pdf               # Informe de la implementación
├── doc/                             # Documentación técnica
├── informe/                         # Análisis académico exhaustivo
└── README.md
```

## Requisitos

- Python 3.7+
- No requiere dependencias externas

## Instalación y Uso

### Ejecución Básica

```bash
# Ejecutar una partida interactiva entre SmartPlayer y RandomPlayer
python main.py
```

### Parámetros Configurables

El archivo `main.py` permite ajustar:
- Tamaño del tablero (por defecto 5×5)
- Tipo de jugadores (se pueden intercambiar)
- Presupuesto temporal por movimiento (por defecto 5 segundos)

### Entregable de Resolución

```bash
python Sebastian_Gonzalez_Alfonso/solution.py
```

## Algoritmos Implementados

### Monte Carlo Tree Search (MCTS) - Base Teórica

MCTS opera mediante iteración de cuatro fases:
1. **Selección**: Descender el árbol usando fórmula UCT hasta nodo no completamente explorado
2. **Expansión**: Agregar nuevo nodo hijo correspondiente a movimiento no explorado
3. **Simulación**: Ejecutar playout aleatorio desde nodo expandido hasta posición terminal
4. **Retropropagación**: Propagar resultado hacia raíz, actualizando estadísticas en cada nodo

Fórmula UCT:
$$\text{UCT}(v) = \frac{Q(v)}{N(v)} + C \sqrt{\frac{\ln N(p(v))}{N(v)}}$$

### Mejoras Implementadas

| Variante | Descripción | Rendimiento |
|----------|-------------|-------------|
| BasicMCTSPlayer | MCTS convencional sin mejoras | Aumenta |
| RecyclerMCTSPlayer | + Reciclaje de árbol entre movimientos | Aumenta |
| RAVEMCTSPlayer | + Evaluación rápida de acciones (RAVE) | Aumenta (máximo) |
| ProgressiveMCTSPlayer | + Parámetros adaptativos por fase | Disminuye |
| ReducedBoardMCTSPlayer | + Análisis de celdas inferiores (fillin) | Sin mejora significativa |
| ParallelizedMCTSPlayer | + Paralelización con sincronización | Aumenta (teoría) |

### Algoritmo Final: SmartPlayer

**Implementación**: Monte Carlo Tree Search paralelizado con las siguientes características:

- **RAVE (Rapid Action Value Estimation)**: Estadísticas all-moves-as-first que aceleran convergencia inicial
- **Reciclaje de árbol**: Reutilización del árbol entre movimientos sucesivos del oponente
- **Union-Find optimizado**: Representación de tablero con verificación de conectividad O(α(n))
- **Sincronización de granularidad fina**: Locks por nodo para minimizar contención
- **Heurísticas de detección temprana**: Identificación rápida de movimientos ganadores/defensivos obvios
- **Heurísticas de Endgame**: Guían la búsqueda en fases avanzadas del juego.

Se recomienda RAVEMCTSPlayer secuencial en CPython, o SmartPlayer en plataformas con verdadero paralelismo (PyPy JIT, Jython, compiladores a máquina nativa).

**Parametrización**:
- Constante de exploración (C): 0.5
- RAVE decay bias: Transición progresiva de RAVE a UCT con visitas
- Threads de worker: 4 (balance entre contención y utilización)

## Análisis Técnico Detallado

Para análisis técnico exhaustivo de algoritmos, estrategias experimentales, comparativas de rendimiento y futura investigación, véase el informe académico:

```
informe/informe.tex
```

## Referencias

- Arneson, B., Hayward, R., Henderson, P. (2010). *Monte-Carlo Tree Search in Hex*
- Cazenave, T. (2009). *Monte-Carlo Hex*
- Chaslot, G. M. J. B., et al. (2008). *Progressive strategies for Monte-Carlo Tree Search*
- Lanctot, M., et al. (2013). *Monte Carlo Tree Search with Heuristic Evaluations using Implicit Minimax Backups*
- Russell, S., Norvig, P. (2009). *Artificial Intelligence: A Modern Approach* (3rd ed.)
- Kunčarová, H., et al. (2019). *Artificial Intelligence for the Hex Game*

## Autor

Sebastian González Alfonso  
MATCOM, Universidad de La Habana  
[sebagonz106@gmail.com](mailto:sebagonz106@gmail.com)

## Licencia

Véase el archivo [LICENSE](LICENSE) para detalles.
