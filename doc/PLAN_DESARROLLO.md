# Plan de Desarrollo del Jugador Virtual de Hex

## Índice
1. [Visión General](#visión-general)
2. [Fases de Desarrollo](#fases-de-desarrollo)
3. [Referencias Bibliográficas](#referencias-bibliográficas)
4. [Consideraciones Técnicas](#consideraciones-técnicas)

---

## Visión General

Este documento describe el plan de desarrollo para crear un jugador virtual competitivo en el juego Hex, implementando técnicas avanzadas de búsqueda y evaluación. El proyecto seguirá un enfoque iterativo, manteniendo funcionalidad mínima en cada paso y mejorando progresivamente la calidad de juego.

**Estrategia bifurcada:**
- **Tableros pequeños (< 7×7):** Alpha-Beta Pruning con MinMax
- **Tableros grandes (≥ 7×7):** Monte Carlo Tree Search (MCTS)

**Herramientas compartidas:**
- Sistema de heurísticas reutilizable
- Framework de evaluación de desempeño
- Utilitarios de optimización y poda

---

## Fases de Desarrollo

### FASE 0: 

**Early-check para victoria inmediata** en `utils/move_checking.py`
   ```python
   def get_immediate_winning_move(board, player_id):
       """
       Detección EFICIENTE de victoria en 1 movimiento
       Complejidad: O(K * N²) donde K ~ 5-20 candidatos << N²
       Tiempo: ~1-2ms tablero 5×5 (EFICIENTE)
       
       Idea: Solo chequear movimientos cercanos a piezas propias,
       no toda la grilla. Una victoria requiere conexión.
       """
       candidates = set()
       for r in range(board.size):
           for c in range(board.size):
               if board.board[r][c] == player_id:
                   for (nr, nc) in board._neighbors(r, c):
                       if board.board[nr][nc] == 0:
                           candidates.add((nr, nc))
       
       for (r, c) in candidates:
           board.place_piece(r, c, player_id)
           if board.check_connection(player_id):
               board.undo_move(r, c)
               return (r, c)
           board.undo_move(r, c)
       
       return None
   
   def get_opponent_forcing_move(board, opp_id):
       """Detectar si oponente gana en 1 movimiento -> BLOQUEAR"""
       return get_immediate_winning_move(board, opp_id)
   ```

---

### FASE 1: Monte Carlo Tree Search Básico (sin heurísticas)
**Duración estimada:** 6-8 horas  
**Prioridad:** ALTA (core functionality)

#### Objetivos
- Implementar MCTS con selección UCT
- Simulaciones totalmente aleatorias en hojas
- Sin persistencia de información entre jugadas
- Arquitectura extensible para futuras mejoras

#### Tareas específicas

1. **Clase `MCTSNode` en `players/mcts_player.py`**
   ```python
   Atributos:
   - board_state: GameState (posición del juego)
   - parent: MCTSNode (referencia al padre)
   - children: dict[tuple, MCTSNode] (movimiento -> nodo hijo)
   - visit_count: int (N(s): número de visitas)
   - win_count: int (V(s): número de victorias)
   - untried_moves: list (movimientos sin explorar aún)
   - player_to_move: int (1 o 2)
   
   Métodos:
   - uct_value(exploration_c): Calcula UCT score
   - is_terminal(): Verifica si es posición terminal
   - select_best_child(): Selecciona hijo con mayor UCT
   ```

2. **Función UCT**
   ```
   UCT(v) = Q(v)/N(v) + C * sqrt(ln(N(parent))/N(v))
   
   Donde:
   - Q(v)/N(v) = promedio de ganancias (exploitation)
   - C = constante de exploración (por defecto 0.1 como especificaste)
   - El segundo término = bonus por exploración
   
   Referencias: "Upper Confidence bound for Trees" (UCB-based strategies)
   en PROGRESSIVE STRATEGIES FOR MONTE-CARLO TREE SEARCH.pdf
   ```

3. **Ciclo de MCTS - Cuatro fases**
   ```
   Repetir N simulaciones:
   
   a) SELECCIÓN (Selection phase):
      - Comenzar en raíz
      - Descender usando UCT hasta nodo no-terminal
      - Nota: Usar por defecto constante C=0.1
   
   b) EXPANSIÓN (Expansion phase):
      - Seleccionar aleatoriamente un movimiento no-explorado
      - Crear nuevo nodo hijo
      - Nota: SIN expansión inmediata en MCTS básico
   
   c) SIMULACIÓN (Simulation/Rollout phase):
      - Desde nodo expandido, jugar aleatoriamente
      - Llenar tablero completamente con asignación aleatoria
      - Sin ninguna heurística en esta fase
   
   d) RETROPROPAGACIÓN (Backpropagation):
      - Actualizar N(v) y V(s) de todos nodos en camino
      - Desde hoja hasta raíz
      - Retropropagar resultado de simulación
   ```

4. **Simulación aleatoria** en `play_random_playout(board_state, starting_player)`
   ```
   Algoritmo:
   - Clonar estado del juego
   - Mientras tablero no esté lleno y no haya ganador:
     - Obtener movimientos legales
     - Seleccionar uno al azar
     - Jugar movimiento
     - Cambiar de jugador
   - Retornar ganador
   ```

5. **Clase `MCTSPlayer` que hereda de `Player`**
   ```python
   Método play(board):
   - Crear raíz con estado actual
   - Ejecutar N simulaciones (por defecto 1000-10000)
   - Seleccionar mejor hijo por visit_count (más visitado)
   - Jugar ese movimiento
   - DESTRUIR árbol (sin persistencia en esta fase)
   - Retornar movimiento
   
   Configurables (en __init__):
   - num_simulations: int
   - exploration_constant: float (por defecto 0.1)
   - max_time: float (segundos, restricción competencia)
   ```

6. **Métodos auxiliares**
   - `get_legal_moves(board)`: Cacheable
   - `clone_game_state(board, history)`: Para rollouts
   - `who_won(board)`: Verificación de ganador
   - `select_action_by_visits(node)`: MAX operación en raíz

#### Algoritmo pseudocódigo MCTS completo
```
function MCTS(root_state, num_simulations, C_explore):
    root ← new MCTSNode(root_state, player=1)
    
    for iteration = 1 to num_simulations:
        node ← root
        state ← clone(root_state)
        
        // SELECTION + EXPANSION
        while not is_terminal(state) and node.visit_count > 0:
            if has_untried_moves(node):
                move ← select_random_untried_move(node)
                play_move(state, move)
                node ← expand_child(node, move)
                break
            else:
                move ← select_by_uct(node, C_explore)
                node ← node.children[move]
                play_move(state, move)
        
        // SIMULATION
        winner ← play_random_playout(state)
        
        // BACKPROPAGATION
        while node != null:
            node.visit_count += 1
            if winner == node.player_to_move:
                node.win_count += 1
            node ← node.parent
    
    // SELECT ACTION: máximo por visit count
    best_move ← argmax over children of root:
                    child.visit_count
    return best_move
```

#### Early-check de victoria (Crítica integrada: one-move winning)

**IMPORTANTE:** Integrar antes de búsqueda MCTS en método `play()`:

```python
class MCTSPlayer(Player):
    def play(self, board):
        # 1. EARLY CHECKS (antes de búsqueda costosa)
        # Costo: O(K*N²) ~ 1-2ms, Ganancia: ~5-10% de jugadas
        
        my_win = get_immediate_winning_move(board, self.player_id)
        if my_win:
            return my_win  # ¡VICTORIA INMEDIATA!
        
        opp_win = get_opponent_forcing_move(board, 3 - self.player_id)
        if opp_win:
            return opp_win  # ¡BLOQUEAR DERROTA!
        
        # 2. MCTS búsqueda normal
        root = MCTSNode(board)
        for _ in range(self.config['num_simulations']):
            self._mcts_iteration(root)
        
        return self._select_best_move(root)
```

**Análisis de costo:**
- Detectar victorias: ~1-2ms (vs O(N⁴) naive → O(K*N²))
- Impacto en tiempo total: <2%
- Aplicable: Ambos MCTS y MinMax

#### Parámetros configurables MCTS v1
```python
{
    'num_simulations': 1000,        # O ajustar por tiempo
    'exploration_constant': 0.1,    # Tu especificación
    'max_time_seconds': 5.0,        # Constraint competencia
    'playout_policy': 'random',     # Solo random en este phase
    'reset_tree_between_moves': True  # Sin persistencia aún
}
```

#### Criteria de éxito
- ✓ MCTSPlayer vs RandomPlayer: MCTS siempre gana
- ✓ Tiempos < 5 segundos por movimiento en tableros 5×5
- ✓ Funcionamiento correcto en tableros de diferentes tamaños
- ✓ Logging detallado de simulaciones

#### Testing recomendado
- Tablero 3×3: Verificar uso completo de árbol
- Tablero 5×5: Benchmark de velocidad
- Comparación estadística: MCTS vs Random (min 100 partidas)

---

### FASE 2: Persistencia Entre Jugadas con Seguridad
**Duración estimada:** 4-5 horas  
**Prioridad:** ALTA (mejora de eficiencia)

#### Referencia bibliográfica
Concepto de "Transposition Tables" de Knuth-Bellman. En MCTS: Reusing information from previous game states.

#### Objetivos
- Reutilizar árbol entre movimientos
- Implementar detección segura de cambios de juego
- Limpieza automática cuando es necesaria
- Pruebas de integridad del árbol

#### Tareas específicas

1. **Modificación de `MCTSPlayer` para persistencia**
   ```python
   Nuevos atributos:
   - self.game_tree: MCTSNode (raíz persistente)
   - self.last_board_hash: str (verificación de integridad)
   - self.tree_reuse_count: int (estadística)
   - self.tree_reset_count: int (estadística)
   ```

2. **Estrategia de detección de cambios**
   ```python
   Algoritmo safe_load_root(new_board, opponent_last_move):
   
   1. Calcular hash de new_board
   2. Si hash != last_board_hash:
      - PROBLEMA: Tablero cambió inesperadamente
      - Rechazar árbol, crear nuevo
      - Registrar en log
   
   3. Si el árbol existe y hash es válido:
      - Buscar nodo correspondiente a opponent_last_move
      - Si existe: usar ese nodo como nueva raíz
      - Si no existe: crear nuevo árbol
      
   4. Casos especiales a detectar:
      a) Ganada previa del oponente: Resetear
      b) Más de un movimiento: Resetear
      c) Primer movimiento de oponente no en árbol: OK (expandir)
   ```

3. **Poda del árbol para memoria**
   ```python
   Función prune_tree(root, max_nodes=100000):
   - Implementar depth-limited search
   - O mejor: path-limited search (limitar profundidad)
   - Eliminar nodos con visit_count muy bajo
   - Mantener nodos más prometedores
   ```

4. **Detección de jugada ganadora**
   ```python
   Método check_winning_condition(board, last_move):
   - Verificar si último movimiento garantiza victoria
   - Si el oponente ganó: resetear árbol
   - Si nosotros podríamos ganar: marcar ramas críticas
   ```

5. **Logging y estadísticas**
   - Registrar reutilizaciones de árbol exitosas
   - Registrar resets forzados
   - Estadística: promedio de nodos reutilizados
   - Estadística: mejora de tiempo por reutilización

#### Código ejemplo - Método principal del jugador
```python
class MCTSPlayer(Player):
    def __init__(self, config):
        self.game_tree = None
        self.last_board_hash = None
        self.config = config
    
    def play(self, board):
        # 1. Intentar reutilizar árbol
        if self.game_tree and self._is_tree_valid(board):
            root = self._navigate_to_board_position(board)
            self.tree_reuse_count += 1
        else:
            root = MCTSNode(board, player=1)
            self.tree_reset_count += 1
        
        # 2. Ejecutar MCTS desde raíz
        for _ in range(self.config['num_simulations']):
            self._mcts_iteration(root)
        
        # 3. Seleccionar acción
        best_move = self._select_best_move(root)
        
        # 4. Persisti árbol
        self.game_tree = root.children[best_move]
        self.last_board_hash = board_to_hash(board)
        
        return best_move
```

#### Criterios de seguridad
- ✗ Nunca propagar información incorrecta
- ✗ Detectar cambios de juego multiplicativos (N movimientos oponente)
- ✓ Reset defensivo cuando hay duda
- ✓ Verificación de hash post-jugada

#### Criteria de éxito
- ✓ MCTSPlayer reutiliza árbol correctamente
- ✓ Detección confiable de cambios
- ✓ Tiempo por movimiento mejora 50-70% con reutilización
- ✓ Sin errores de estado en 1000+ partidas

#### Testing recomendado
- Alter board between moves → árbol resetea
- Opponent plays as expected → árbol reutiliza
- Compare move quality with/without persistence

---

### FASE 3: Rapid Action Value Estimation (RAVE)
**Duración estimada:** 5-6 horas  
**Prioridad:** MEDIA-ALTA (mejora de convergencia)

#### Referencia bibliográfica
**"Monte Carlo Tree Search in Hex.pdf"**: RAVE (Rapid Action Value Estimation) de Gelly & Silver (2011). Concepto: All-Moves-As-First (AMAF) statistics.

**"Monte Carlo Tree Search with Heuristic Evaluations.pdf"**: Aplicación de RAVE con evaluaciones heurísticas post-playout.

#### Objetivos
- Implementar RAVE statistics en cada nodo
- Combinar UCT + RAVE con factor de decaimiento
- Mejorar velocidad de convergencia en primeras iteraciones
- Investigar balance óptimo entre UCT y RAVE

#### Concepto de RAVE
```
En MCTS tradicional:
- Solo se actualiza el nodo cuando está EN el camino
- Primeros movimientos tardan muchas simulaciones en converger

Con RAVE:
- Se actualiza TODO movimiento que aparece en el playout
- Aunque no estuviera en nuestro árbol principal
- Proporciona estimación rápida de valor de acción

Notación:
- N_AMAF(a): número de veces que acción 'a' fue jugada en playouts
- Q_AMAF(a): rewards acumulados de acción 'a'

Fórmula de selección mejorada:
UCB(v) = (1-β) * UCT(v) + β * RAVE(v)

Donde β decae con visit_count:
β(N(v)) = sqrt(threshold / (3 + N(v)))  [típicamente threshold=1000-10000]
```

#### Tareas específicas

1. **Modificación de `MCTSNode` para RAVE**
   ```python
   Nuevos atributos:
   - rave_win_count: dict[move] → int
   - rave_visit_count: dict[move] → int
   - amaf_stats: dict[move] → {'Q': float, 'N': int}
   ```

2. **Actualización de playouts para RAVE**
   ```python
   Función play_and_track_amaf(initial_state, initial_player):
   - Ejecutar playout aleatorio normal
   - Registrar ALL movimientos jugados
   - Registrar TODOS los resultados de movimientos
   - Retornar: (winner, moves_sequence, amaf_rewards)
   
   Nota: Esto es más costoso que random playout,
   pero la mejora lo vale. Ver "MCTS in Hex.pdf" pp. X-X
   ```

3. **Backpropagation mejorado**
   ```python
   Modificar backprop para actualizar RAVE:
   
   while node != null:
       // UCT update (tradicional)
       node.visit_count += 1
       if winner == node.player_to_move:
           node.win_count += 1
       
       // RAVE update (NUEVO)
       for move in amaf_sequence:
           if move not in node.rave_visit_count:
               node.rave_visit_count[move] = 0
               node.rave_win_count[move] = 0
           node.rave_visit_count[move] += 1
           if winner == node.player_to_move:
               node.rave_win_count[move] += 1
       
       node = node.parent
   ```

4. **Función de selección RAVE**
   ```python
   def uct_rave_value(node, move, exploration_c, rave_threshold=1000):
       N = node.visit_count
       
       // Componente UCT
       if move in node.children:
           child = node.children[move]
           uct_val = (child.win_count / child.visit_count) + \
                     exploration_c * sqrt(ln(N) / child.visit_count)
       else:
           uct_val = float('inf')  # Unexplored bonus
       
       // Componente RAVE
       if move in node.rave_visit_count and node.rave_visit_count[move] > 0:
           amaf_val = node.rave_win_count[move] / node.rave_visit_count[move]
           beta = sqrt(rave_threshold / (3 + N))
           rave_val = (1 - beta) * uct_val + beta * amaf_val
       else:
           rave_val = uct_val
       
       return rave_val
   
   Referencias: Gelly & Silver (2011) "Monte-Carlo Tree Search with Rapid
   Action Value Estimation" - IEEE TAI
   ```

5. **Configuración de RAVE**
   ```python
   {
       'rave_enabled': True,
       'rave_threshold': 1000,      # Punto donde β → 0
       'beta_decay': 'sqrt',         # Función de decaimiento
       'track_all_amaf': True,       # Rastrear todos movimientos
       'rave_bias_conservative': 0.3  # Factor conservador
   }
   ```

#### Comparativas teóricas a investigar
- Ver sección "RAVE for Hex" en **"Monte Carlo Tree Search in Hex.pdf"**
- Comparación: UCT puro vs RAVE vs combinación
- Trade-off entre costo computacional vs convergencia

#### Resultados esperados
```
Benchmark (tablero 5×5, 10000 simulaciones):
- MCTS básico: 1-2 segundos
- MCTS + RAVE: 1.2-2.5 segundos (overhead de tracking)
- Mejora de win_rate vs Random: +5-15% (por convergencia más rápida)
```

#### Criteria de éxito
- ✓ RAVE stats correctamente acumuladas
- ✓ Win rate mejorado vs MCTSv1 en tableros grandes
- ✓ Convergencia más rápida observada
- ✓ Impacto mínimo en tiempo de ejecución

#### Testing recomendado
- Comparar MCTSv2 vs MCTSv1 en mismas condiciones (mismo seed)
- Verificar que RAVE values convergen a valores razonables
- Tableros 4×4: Debug visual de stats RAVE

---

### FASE 4: Estrategias Progresivas (Progressive Strategies)
**Duración estimada:** 7-8 horas  
**Prioridad:** MEDIA (mejora de sesgo de búsqueda)

#### Referencia bibliográfica
**"PROGRESSIVE STRATEGIES FOR MONTE-CARLO TREE SEARCH.pdf"**: Huang et al. estrategia de progressive widening y progressive unpruning.

Concepto fundamental: En MCTS, al principio del juego hay muchas opciones abiertas. Al final, menos. La estrategia debe adaptarse:
- **Temprano:** Favorecer exploración agresiva (más anchura en árbol)
- **Tarde:** Favorecer explotación (profundizar en ramas prometedoras)

#### Objetivos
- Implementar progressive widening (limitar expansión inicial)
- Implementar progressive unpruning (abrir nodos limitados)
- Implementar estrategia progresiva alternativa: RAVE mejorado con bias
- Comparar ambas estrategias en partidas reales

#### Concepto de Progressive Widening
```
Idea: No expandir todos los movimientos inmediatamente.

En inicio: Expandir solo k movimientos
Cuando N(parent) crece: Liberar más movimientos gradualmente

Fórmula típica:
k(t) = C * t^p

Donde:
- t = tiempo o iteraciones
- C, p = constantes (típicamente p=2/3 para Hex)
- Resultado: Primeros movimientos siempre evaluados,
  luego nuevos movimientos se abren progresivamente

Beneficio: Profundiza en líneas prometedoras antes de
explorar todas las opciones superficialmente.
```

#### Tareas específicas

1. **Implementar Progressive Widening (PW)**
   ```python
   Clase MCTSProgressiveNode(MCTSNode):
   
   Nuevos atributos:
   - expansion_count: int (cuántos movimientos se han abierto)
   - max_expansion: calculated dinámicamente
   
   Método expansion_limit(visit_count):
   """Retorna cuántos movimientos pueden estar abiertos ahora"""
   c = 1.0  # Constante configurable
   p = 2/3  # Exponente para Hex
   return int(c * (visit_count ** p))
   
   Comparación vs expected: Ver tabla en
   "PROGRESSIVE STRATEGIES FOR MONTE-CARLO TREE SEARCH.pdf", Fig 3
   ```

2. **Fase de expansión mejorada**
   ```python
   Método select_move_for_expansion():
   current_expansion = self.expansion_limit(self.visit_count)
   
   if len(self.children) < current_expansion:
       // Permitir expandir más movimientos
       untried = [m for m in self.untried_moves 
                  if m not in self.children]
       return random.choice(untried)
   else:
       // Ya hemos expandido suficiente, solo seleccionar entre abiertos
       return select_by_uct(self.children)
   ```

3. **Implementar Alternative Progress Strategy**
   
   Basada en idea de **"Progressive Strategies using RAVE boost"**:
   
   ```python
   Estrategia alternativa: RAVE con sesgo progresivo
   
   La idea: En inicio del búsqueda, RAVE domina (exploración amplia)
           Conforme crece el árbol, UCT dominDa (explotación)
   
   Fórmula mejorada:
   β_progressive(N) = sqrt(RAVE_THRESHOLD / (1 + N)) * decay_factor
   
   Donde decay_factor = 1 - (iteration / total_iterations)
   
   Efecto: Exploración uniforme → progresivamente más selectiva
   ```

4. **Comparación de estrategias**
   
   Implementar clase `MCTSComparator`:
   ```python
   def compare_strategies(board_size, num_games, num_sims):
       players = {
           'vanilla_mcts': MCTSPlayer({'strategy': 'vanilla'}),
           'pw_mcts': MCTSPlayer({'strategy': 'progressive_widening'}),
           'rave_progressive': MCTSPlayer({'strategy': 'rave_progressive'})
       }
       
       resultados = {}
       for p1_name, p1 in players.items():
           for p2_name, p2 in players.items():
               if p1_name < p2_name:  # Evitar duplicates
                   win_rate = evaluate_match(p1, p2, num_games)
                   resultados[(p1_name, p2_name)] = win_rate
       
       return resultados
   ```

5. **Parámetros de configuración**
   ```python
   Progressive Widening:
   {
       'pw_enabled': True,
       'pw_constant': 1.0,           # C en k(t) = C*t^p
       'pw_exponent': 2/3,           # p en fórmula
   }
   
   RAVE Progressive:
   {
       'rave_progressive': True,
       'rave_progressive_decay': 'linear',  # o 'sqrt'
       'initial_rave_bias': 0.8,    # β(0) = 0.8
       'final_rave_bias': 0.0,      # β(∞) → 0.0
   }
   ```

#### Análisis teórico esperado
```
Referencias directas a "PROGRESSIVE STRATEGIES...pdf":

Section 3.2: Progressive Widening en juegos con factor de rama alto
- Hex tiene factor de rama promedio: N² (tablero N×N)
- PWxecuta reduce carga computacional en raíz
- Expectativa: 20-30% mejora en win_rate vs vanilla

Section 4: Progressive Unpruning
- Mover de búsqueda ancha a profunda
- Propiedades teóricas de convergencia preservadas
- Garantías de optimalidad asintótica

Implementar exactamente como se describe en sección 5.1
para Hex.
```

#### Criteria de éxito
- ✓ Progressive Widening reduce nodos iniciales
- ✓ RAVE Progressive muestra sesgo temporal esperado
- ✓ Win rate de PW vs vanilla en rangos teóricos
- ✓ Ambas estrategias convergen a misma política final

#### Testing recomendado
- Visualizar árbol de expansión progresiva
- Comparar (vanilla, PW, RAVE_prog) en 50+ partidas
- Analizar profundidad promedio vs anchura

---

### FASE 5: Comparación de Criterios de Selección de Jugadas
**Duración estimada:** 4-5 horas  
**Prioridad:** MEDIA (análisis comparativo)

#### Objetivos
- Implementar múltiples criterios de selección final
- Comparar con método tradicional (máximo visit_count)
- Investigar impacto en win_rate y exploración

#### Criterios a implementar

1. **Criterio tradicional (Baseline)**
   ```python
   def select_by_max_visits(root_node):
       """Soleccionar hijo con máximas visitas"""
       return argmax(children: child.visit_count)
   
   Fundamento: Más visitados = mejor estimado por ley de grandes números
   
   Ventajas: Simple, robusto
   Desventajas: Puede no reflejar win_rate real si exploración sesgada
   ```

2. **Criterio de win_rate puro**
   ```python
   def select_by_max_win_rate(root_node):
       """Seleccionar hijo con máximo Q(v)/N(v)"""
       return argmax(children: child.win_count / child.visit_count)
   
   Fundamento: Explota las evaluaciones más exitosas
   
   Ventajas: Directamente optimiza win_rate
   Desventajas: Puede explorar poco ciertas ramas, sesgo estadístico
   ```

3. **Criterio propuesto: win_rate + visits/total_sims**
   ```
   TU IDEA: Combinar:
   - wins/plays (win_rate = Q(v)/N(v))
   - plays/total_simulations (relative exploration)
   
   Fórmula sugerida:
   score(v) = α * (Q(v)/N(v)) + (1-α) * (N(v) / Σ N(children))
   
   Donde α ∈ [0,1] es balance exploitation/exploration
   
   Interpretación:
   - α=1: Pure win_rate (high variance early)
   - α=0: Pure visit count (robust but slower)
   - α=0.7: Balanced (recomendación)
   
   Ventaja: Combina solidez estadística + optimalidad esperada
   ```

4. **Criterio UCB-based**
   ```python
   def select_by_uct_root(root_node, exploration_c):
       """Aplicar UCT formula a nivel raíz"""
       return argmax(children: child_uct_score(exploration_c))
   
   Ventaja: Continua mismo criterio de selección internamente
   Desventaja: Factor de exploración puede no ser óptimo post-búsqueda
   ```

5. **Criterio híbrido propuesto (Investigación)**
   ```
   Idea: Seleccionar según confianza estadística
   
   score(v) = Q(v)/N(v) + confidence_interval(v, confidence=0.95)
   
   Interpretación: Seleccionar mejor estimado si confiamos en él,
   sino el que más hemos explorado
   
   Referencias: Confidence intervals in bandit problems (Kaufmann et al)
   ```

#### Tareas específicas

1. **Crear módulo `selection_strategies.py`**
   ```python
   class SelectionStrategy(ABC):
       @abstractmethod
       def select(self, node: MCTSNode) -> tuple:
           """Retorna (selected_move, score_info)"""
           pass
   
   class MaxVisitsStrategy(SelectionStrategy): ...
   class MaxWinRateStrategy(SelectionStrategy): ...
   class HybridStrategy(SelectionStrategy): ...
   # etc
   ```

2. **Integración en MCTSPlayer**
   ```python
   mcts_player = MCTSPlayer(config={
       'selection_strategy': 'hybrid',  # Configurable
       'hybrid_alpha': 0.7,
   })
   ```

3. **Benchmark comparativo**
   ```python
   def benchmark_selection_strategies(board_size, num_games):
       strategies = [
           ('max_visits', MaxVisitsStrategy()),
           ('max_winrate', MaxWinRateStrategy()),
           ('hybrid_07', HybridStrategy(alpha=0.7)),
           ('uct_root', UCTRootStrategy()),
       ]
       
       results = {}
       for name, strat in strategies:
           player = MCTSPlayer(selection_strategy=strat)
           win_rate = evaluate_vs_random(player, num_games)
           results[name] = win_rate
       
       return results
   ```

#### Criteria de éxito
- ✓ Todos los criterios implementados y probados
- ✓ Benchmarks generados para todas las estrategias
- ✓ Diferencias significativas detectadas
- ✓ Mejor estrategia identificada

#### Resultados esperados
```
En tablero 5×5 con 10k simulaciones vs Random:
- max_visits: ~90% win_rate (baseline)
- max_winrate: ~92% (más agresivo)
- hybrid_0.7: ~93% (balanceado)
- uct_root: ~88% (exploración continua)
```

---

### FASE 6: Heurísticas de Evaluación del Juego
**Duración estimada:** 8-10 horas  
**Prioridad:** ALTA (core improvement)

#### Referencia bibliográfica
**"Monte Carlo Tree Search with Heuristic Evaluations.pdf"**: Integration de heurísticas en playouts.
**"Artificial Intelligence for the Hex Game.pdf"**: Specific Hex heuristics (connectivity, distance, etc).

**NOTA CRÍTICA (Crítica 1 integrada):** Se ha preparado documento complementario **`PATRONES_HEX_DETECCION.md`** 
con implementaciones EFICIENTES de patrones específicos (bridges, ziggurats, etc). 
Este documento resuelve la crítica sobre patrones "genéricos" → concretos.
**DEBE consultarse al implementar heurísticas.**

#### Objetivos
- Diseñar heurísticas de evaluación rápidas
- Implementar detección eficiente de patrones (Bridge, Ziggurat, etc)
- Integrar en playouts de MCTS
- Mejorar calidad de simulaciones
- Comparar impacto en convergencia

#### Heurísticas a implementar

1. **Heurística 1: Distance to Winning (DTW)** - 40% del score
   ```
   Concepto: Conectividad relativa al objetivo
   - Jugador 1: distancia mínima left-to-right
   - Jugador 2: distancia mínima top-to-bottom
   
   Implementación:
   - Usar BFS para encontrar componentes conectadas
   - Calcular distancia mínima de componente a borde opuesto
   - Score = 1 / (1 + distancia), normalizado
   
   Complejidad: O(N²) BFS
   
   Ver: "Artificial Intelligence for the Hex Game.pdf",
   Section "Heuristic Evaluation Functions"
   ```

2. **Heurística 2: Connectivity Analysis** - 20% del score
   ```
   Implementación mejorada: Usar Union-Find para O(N² α(N))
   
   CRÍTICO: Ver PATRONES_HEX_DETECCION.md, sección 2
   para implementación de ConnectivityAnalyzer class
   
   - Detectar largest component size
   - Detectar winning path (borde a borde opuesto)
   - Caching seguro para reutilización
   
   score = largest_component_size / (board.size²)
   ```

3. **Heurística 3: Pattern-Based Territory** - 25% del score (NUEVA - Crítica 1)
   ```
   Detección eficiente de formaciones ganadoras específicas:
   
   a) BRIDGE (Puente - defensa/ataque crítico)
      - Dos piezas propias a distancia 2
      - Garantizan conexión (oponente NO puede bloquear ambas)
      - Detección: O(1) amortizado → O(N²) total
      - Bonus: +30 por bridge
      
      IMPLEMENTACIÓN: Ver PATRONES_HEX_DETECCION.md, sección 1
      Función: detect_bridges(board, player_id)
   
   b) ZIGGURAT (Estructura densa - territorio)
      - Cluster de piezas con alta densidad local (3+ vecinos propios)
      - Difícil de atacar
      - Detección: O(N²) cacheado por turno
      - Bonus: variable según tamaño/densidad
      
      IMPLEMENTACIÓN: Ver PATRONES_HEX_DETECCION.md, sección 3
      Función: detect_ziggurats(board, player_id)
   
   Tiempo TOTAL patrones: ~40-50ms/turno (UNA SOLA VEZ, CACHEADO)
   Overhead en MCTS: <5%
   Win rate improvement: +8-15%
   ```

4. **Heurística Compuesta: Evaluation Function Integrada** (MEJORADA - Crítica 1)
   ```python
   def evaluate_hex_position_comprehensive(board, player_id):
       """
       Evaluación integrada con detección de patrones.
       Tiempo cacheado: ~50ms por turno (ACCEPTABLE)
       Overhead por simulación: ~1ms promedio
       """
       
       # Componentes rápidas (CACHEAR UNA SOLA VEZ)
       bridges = detect_bridges(board, player_id)  # [PATRONES_HEX_DETECCION.md]
       connectivity = ConnectivityAnalyzer(board, player_id)  # [PATRONES_HEX_DETECCION.md]
       ziggurats = detect_ziggurats(board, player_id)  # [PATRONES_HEX_DETECCION.md]
       
       # Scoring multi-componente
       score = 0.0
       
       # 1. Distance to Winning (40%)
       dtw = compute_distance_to_winning(board, player_id)
       score += 0.4 * dtw
       
       # 2. Bridges (25%)
       bridge_count = len(bridges)
       bridge_strength = bridge_count * (1.0 if bridge_count >= 3 else 0.5)
       score += 0.25 * bridge_strength
       
       # 3. Connectivity (20%)
       largest_comp = connectivity.get_largest_component()
       comp_ratio = largest_comp / (board.size ** 2 / 2)
       score += 0.20 * comp_ratio
       
       # 4. Territory/Ziggurats (15%)
       ziggurat_bonus = sum(z['score'] for z in ziggurats)
       score += 0.15 * min(ziggurat_bonus, 10)
       
       # Bonus decisivo por winning path
       if connectivity.is_winning_path_exists():
           score += 0.5  # Bonus HUGE
       
       # Normalizar [-1, 1]
       return min(1.0, max(-1.0, (score / 10.0)))
   
   # Impacto esperado en MCTS:
   # - Sin patrones: ~85% win_rate vs Random
   # - Con patrones: ~93% win_rate vs Random (+8%)
   # - Tiempo: +40ms UNA SOLA VEZ al inicio del turno
   ```

#### Tareas específicas

1. **Crear archivo `heuristics/hex_heuristics.py`**
   ```python
   class HexHeuristic(ABC):
       @abstractmethod
       def evaluate(self, board: HexBoard, player_id: int) -> float:
           pass
   
   class DistanceToWinningHeuristic(HexHeuristic): ...
   class ConnectivityHeuristic(HexHeuristic): ...
   class TerritoryHeuristic(HexHeuristic): ...
   
   class CompositeHeuristic(HexHeuristic):
       def __init__(self, heuristics: list, weights: list):
           self.heuristics = heuristics
           self.weights = weights
   ```

2. **Integración en playouts**
   ```python
   Opción A: Bias de políticas
   - Al jugar movimiento en playout, preferir alto-evaluados
   - Mantener randomness (probabilidad κ)
   
   Opción B: Terminal evaluation
   - Al fin del playout, retornar eval en vez de winner binary
   - Proporciona gradual information
   
   Recomendación: Opción A por eficiencia
   Referencias: "Heuristic Evaluations in MCTS.pdf", Section 4.2
   ```

3. **Heurística-guided playout policy**
   ```python
   def play_heuristic_playout(board, player, heuristic, kappa=0.3):
       """Playout con sesgo heurístico"""
       state = board.clone()
       current_player = player
       
       while not is_terminal(state):
           moves = get_legal_moves(state)
           
           if random.random() < kappa:
               // Pure random (exploration)
               move = random.choice(moves)
           else:
               // Heuristic-biased
               scores = [heuristic.evaluate_move(state, m, current_player)
                         for m in moves]
               // Softmax selection
               probs = softmax(scores)
               move = np.random.choice(moves, p=probs)
           
           state.place_piece(*move, current_player)
           current_player = 3 - current_player
       
       return determine_winner(state)
   ```

4. **Benchmarking de heurísticas**
   ```python
   Comparar:
   - MCTS puro (random playouts) vs MCTS + heurística
   - Metrics:
     - Win rate vs Random
     - Win rate vs MCTS puro
     - Convergence speed (win_rate by # simulations)
   
   Expectedresults:
   - MCTS + heurística: +15-25% faster convergence
   - Win rate improvement: +5-10%
   
   References: Figure 7 "MCTS with Heuristic Evaluations.pdf"
   ```

5. **Tuning de pesos**
   ```python
   Evaluación de configuraciones:
   w_dtw ∈ {0.3, 0.5, 0.7}
   w_conn ∈ {0.2, 0.3, 0.5}
   w_terr ∈ {0.0, 0.1, 0.2}
   
   Usar grid search con tournament
   ```

#### Heurística recomendada (Custom para Hex)
```
Basada en literatura y experiencia:
- Distance to Winning: Métrica primaria (50%)
- Territory: Patrones de puentes (30%)
- Defensa: Movimientos de bloqueo (20%)

Implementación agresiva:
- Detectar movimientos perdedores (oponente gana en 1): -∞
- Detectar movimientos ganadores (nosotros ganamos en 1): +∞
- Otros: scoring heurístico

Ver "Artificial Intelligence for the Hex Game.pdf", 
Capítulo sobre strategy evaluation
```

#### Criteria de éxito
- ✓ Heurísticas evaluadas y comparadas
- ✓ Mejor combinación identificada
- ✓ Win rate vs Random mejorado significativamente
- ✓ Código modular y reutilizable (para MinMax)

---

### FASE 7: Algoritmo MinMax con Alpha-Beta Pruning (tableros pequeños)
**Duración estimada:** 7-9 horas  
**Prioridad:** ALTA (para tableros < 7×7)

#### Referencia bibliográfica
Algoritmos clásicos en "Artificial Intelligence: A Modern Approach" (Russell & Norvig).
**"Artificial Intelligence for the Hex Game.pdf"**: Aplicación específica a Hex.

#### Objetivos
- Implementar MinMax con alpha-beta pruning
- Reutilizar heurísticas de evaluación (Fase 6)
- Profundidad adaptativa según complejidad
- Comparar con MCTS en tableros pequeños

#### Algoritmo MinMax con Alpha-Beta

```
function MiniMax(position, depth, alpha, beta, maximizing_player):
    if depth == 0 or is_terminal(position):
        return (evaluate_heuristic(position), null)
    
    if maximizing_player:
        maxEval = -∞
        best_move = null
        for each move in legal_moves(position):
            new_pos = make_move(position, move)
            (eval, _) = MiniMax(new_pos, depth-1, alpha, beta, FALSE)
            
            if eval > maxEval:
                maxEval = eval
                best_move = move
            
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  // Beta cutoff
        
        return (maxEval, best_move)
    else:
        minEval = +∞
        best_move = null
        for each move in legal_moves(position):
            new_pos = make_move(position, move)
            (eval, _) = MiniMax(new_pos, depth-1, alpha, beta, TRUE)
            
            if eval < minEval:
                minEval = eval
                best_move = move
            
            beta = min(beta, eval)
            if beta <= alpha:
                break  // Alpha cutoff
        
        return (minEval, best_move)

// Llamada inicial
(score, best_move) = MiniMax(root, MAX_DEPTH, -∞, +∞, TRUE)
return best_move
```

#### Optimizaciones

1. **Move Ordering (crítico para poda)**
   ```
   Idea: Evaluar primero movimientos prometedores
        minimiza desorden en tree, maximiza cutoffs
   
   Implementing:
   - Usar heurística para ordenar movimientos
   - Movimientos recientes mejores primero (history heuristic)
   - Movimientos de capítulo anterior primero (killer heuristic)
   
   Impacto: 50-70% speedup en alpha-beta típico
   
   Para Hex: Ordenar por heurística de "distancia a victoria"
   ```

2. **Transposition Tables (Memoization)**
   ```
   Concepto: Guardar resultados de posiciones vistas
   
   Implementation:
   - Usar board hash como clave
   - Guardar (depth, score, type) donde type ∈ {EXACT, LOWER, UPPER}
   - Verificar antes de evaluar
   
   Impacto: 30-50% consultas pueden reutilizarse
   
   Referencias: "Alpha-Beta Pruning with Memory" (Knuth & Moore)
   ```

3. **Iterative Deepening**
   ```
   Idea: En vez de búsqueda fija depth-D, hacer depth-1, depth-2, ..., depth-D
        hasta que se acabe tiempo
   
   Ventaja: Garantizado encontrar mejor solución con tiempo disponible
            Transposition table crece gradualmente
   
   Complejidad: O(b^d) vs O(b^(d+1)) pero mejor constant factors
   
   Implementación pseudocódigo:
   
   for depth = 1 to MAX_DEPTH:
       if time_expired():
           return best_move_so_far
       (score, move) = MiniMax(board, depth, -∞, +∞, maximizing)
       best_move = move
       if score es victoria garantizada:
           return move
   ```

#### Tareas específicas

1. **Crear `players/minimax_player.py`**
   ```python
   class MinimaxPlayer(Player):
       def __init__(self, config):
           self.max_depth = config.get('max_depth', 6)
           self.heuristic = config.get('heuristic')  # Fase 6
           self.transposition_table = {}  # Zobrist hashing
           self.time_limit = config.get('max_time', 5.0)
           self.move_order_heuristic = ...
   
       def play(self, board):
           // Iterative deepening con time limit
           best_move = None
           for depth in range(1, self.max_depth + 1):
               if self.time_exceeded():
                   break
               (score, move) = self.minimax(board, depth, -inf, inf, True)
               best_move = move
               
               // Early exit si victoria forzada encontrada
               if score > WINNING_THRESHOLD:
                   break
           
           return best_move
       
       def minimax(self, board, depth, alpha, beta, maximizing):
           // Verificar transposition table
           board_hash = self.get_board_hash(board)
           if board_hash in self.transposition_table:
               entry = self.transposition_table[board_hash]
               if entry['depth'] >= depth:
                   return entry['value']
           
           // Terminal check
           if depth == 0 or board.is_full():
               eval = self.heuristic.evaluate(board, self.player_id)
               return (eval, None)
           
           // Check win condition
           if board.check_connection(1):
               return (1.0 if self.player_id == 1 else -1.0, None)
           if board.check_connection(2):
               return (-1.0 if self.player_id == 1 else 1.0, None)
           
           // Recursive minimax
           if maximizing:
               max_eval = -inf
               best_move = None
               for move in self.order_moves(board):
                   new_board = board.clone()
                   new_board.place_piece(*move, self.player_id)
                   (eval, _) = self.minimax(new_board, depth-1, alpha, beta, False)
                   
                   if eval > max_eval:
                       max_eval = eval
                       best_move = move
                   alpha = max(alpha, eval)
                   if beta <= alpha:
                       break
           else:
               // Similar para minimizing
               ...
           
           // Store en transposition table
           self.transposition_table[board_hash] = {
               'value': (max_eval, best_move),
               'depth': depth
           }
           
           return (max_eval, best_move)
       
       def order_moves(self, board):
           """Ordenar movimientos por promesa heurística"""
           moves = get_legal_moves(board)
           # Score cada movimiento temporalmente
           scored = [(self.move_score(board, m), m) for m in moves]
           scored.sort(reverse=True)
           return [m for _, m in scored]
   ```

2. **Función heurística de evaluación** (Reutilizar Fase 6)
   ```python
   def evaluate_position_for_minimax(board, player_id):
       """
       Scale from [-1, 1]:
       1.0 = nosotros ganamos
       -1.0 = oponente gana
       0.5 = favorable
       0.0 = neutral
       """
       
       // Check terminal states
       if board.check_connection(player_id):
           return 1.0
       if board.check_connection(3 - player_id):
           return -1.0
       
       // Use Phase 6 heuristics
       score = self.composite_heuristic.evaluate(board, player_id)
       
       // Adjust for game progression (más profundo = menos cambio)
       emptiness = num_empty_cells / board.size**2
       uncertainty = emptiness * 0.3  # Uncertainty bonus
       
       return score + uncertainty
   ```

3. **Transposition Table con Zobrist Hashing**
   ```python
   class TranspositionTable:
       def __init__(self):
           self.table = {}
           self.zobrist_random = {}
           self.init_zobrist()
       
       def init_zobrist(self):
           """Inicializar números aleatorios para hashing"""
           random.seed(42)  # Reproducible
           for pos in range(BOARD_MAX_SIZE ** 2):
               for player in [1, 2]:
                   self.zobrist_random[(pos, player)] = random.getrandbits(64)
       
       def compute_hash(self, board):
           """Computa Zobrist hash de posición"""
           hash_val = 0
           for r in range(board.size):
               for c in range(board.size):
                   if board.board[r][c] != 0:
                       pos = r * board.size + c
                       hash_val ^= self.zobrist_random[(pos, board.board[r][c])]
           return hash_val
       
       def store(self, board, depth, score, flag):
           """flag ∈ {EXACT, LOWER, UPPER}"""
           hash_val = self.compute_hash(board)
           self.table[hash_val] = {
               'score': score,
               'depth': depth,
               'flag': flag
           }
       
       def lookup(self, board, depth):
           """Retorna (encontrado, score) o (False, None)"""
           hash_val = self.compute_hash(board)
           if hash_val in self.table:
               entry = self.table[hash_val]
               if entry['depth'] >= depth:
                   return (True, entry['score'])
           return (False, None)
   ```

4. **Benchmark MinMax vs MCTS**
   ```python
   def compare_minmax_vs_mcts(board_size):
       minimax = MinimaxPlayer({'max_depth': 6, 'heuristic': composite})
       mcts = MCTSPlayer({'num_simulations': 10000, 'rave': True})
       
       // Tableros pequeños
       if board_size <= 4:
           minimax should win > 70%
       elif board_size == 5:
           minimax should be comparable
       elif board_size >= 6:
           mcts should start winning (branching factor too high)
   ```

#### Criteria de éxito
- ✓ MinMax implementado con alpha-beta working
- ✓ Transposition table funcional
- ✓ Superiority vs MCTS en tableros 3×3, 4×4, 5×5
- ✓ Tiempo por movimiento < 5 segundos
- ✓ Heurísticas correctamente reutilizadas

#### Resultados esperados
```
Tablero 3×3, depth=9:
- MinMax: Victoria garantizada (agota búsqueda completa)
- Tiempo: <0.1 segundo

Tablero 4×4, depth=6:
- MinMax vs Random: ~99% win_rate
- MinMax vs MCTS (same time): ~60-70% win_rate
- Tiempo: 1-3 segundos

Tablero 5×5, depth=4:
- MinMax vs MCTS (10k sims): ~50-55% win_rate
- Tiempo: 3-5 segundos
```

---

### FASE 8: Optimización, Poda y Ajustes Finales
**Duración estimada:** 6-8 horas  
**Prioridad:** MEDIA (performance tuning)

#### Objetivos
- Identificar y optimizar cuellos de botella específicos de Hex
- Implementar strategic pruning del árbol MCTS
- Perfilado detallado del código
- Preparar benchmarks finales

#### Cuellos de Botella Específicos de Hex (Crítica 3 integrada)

**Distribución de tiempo en MCTS típico durante simulaciones:**
```
50-60%: Playouts (simulación aleatoria)
20-25%: Tree traversal + backpropagation  
10-15%: Move generation (get_legal_moves)
 5-10%: Terminal checks (check_connection)
```

**CRITICIDAD ULTRA-ALTA (50% del tiempo): get_legal_moves()**
- Llamado 1000s de veces por búsqueda MCTS
- Implementación naive iterando grilla: O(N²) cada vez = INACEPTABLE
- **SOLUCIÓN OBLIGATORIA (invertir 1-2 horas):**
  ```python
  class OptimizedGameState:
      def __init__(self, board):
          # Bitmap de celdas vacías (set o bitarray)
          self.empty_positions = {(r, c) for r in range(board.size)
                                  for c in range(board.size)
                                  if board.board[r][c] == 0}
          self.empty_count = len(self.empty_positions)
      
      def get_legal_moves_fast(self):
          """O(empty_count) ~ O(N²/2) en inicio, O(1) después"""
          return list(self.empty_positions)
      
      def place_piece_fast(self, r, c, player_id):
          if (r, c) in self.empty_positions:
              self.empty_positions.discard((r, c))
              self.empty_count -= 1
              return True
          return False
  ```
  **Ganancia esperada:** 40-50% reducción de tiempo total

**CRITICIDAD ALTA (10-15% del tiempo): check_connection()**
- Usado al FIN de cada playout (1000s de veces)
- Implementación naive: BFS completo O(N²) siempre
- **SOLUCIÓN OBLIGATORIA (invertir 1-2 horas):**
  ```python
  def check_connection_fast(board, player_id):
      """BFS con early-exit cuando se detecta victoria"""
      visited = set()
      queue = deque()
      
      if player_id == 1:
          # Buscar path left-to-right
          for r in range(board.size):
              if board.board[r][0] == 1:
                  queue.append((r, 0))
                  visited.add((r, 0))
          
          while queue:
              r, c = queue.popleft()
              if c == board.size - 1:
                  return True  # EARLY EXIT: ¡ENCONTRADA!
              
              for (nr, nc) in board._neighbors(r, c):
                  if (nr, nc) not in visited and board.board[nr][nc] == 1:
                      visited.add((nr, nc))
                      queue.append((nr, nc))
      else:
          # Similar para player 2 (top-to-bottom)
          ...
      
      return False
  ```
  **Ganancia esperada:** 30-40% reducción en check_connection
  **Razón:** Termina BFS apenas encuentra victoria (no explora resto del tablero)

**CRITICIDAD MEDIA (20-25%): Backpropagation en árbol**
- Recorre pathroot-leaf (típicamente 20-30 nodos por playout)
- Operación O(depth) repetida 1000s de veces
- **SOLUCIÓN RECOMENDADA (invertir 30 min):**
  ```python
  class OptimizedMCTSNode:
      __slots__ = ['parent', 'visit_count', 'win_count', 'children', 'untried_moves']
      
      def __init__(self, parent=None):
          self.parent = parent  # Traversal directo sin BFS
          self.visit_count = 0
          self.win_count = 0
          self.children = {}
          self.untried_moves = []
      
      def backpropagate_fast(self, reward):
          """Actualización O(depth) sin overhead BFS"""
          node = self
          while node is not None:
              node.visit_count += 1
              node.win_count += reward
              node = node.parent
  ```
  **Uso de __slots__:** Reduce overhead por instancia de ~200-300 bytes a ~80-100 bytes
  **Ganancia esperada:** 10-15% reducción + mejor memory footprint

#### Tareas específicas

1. **Optimizaciones OBLIGATORIAS (Crítica 3)** 
   - [ ] Aplicar `get_legal_moves()` optimizado con set
   - [ ] Aplicar `check_connection()` con early-exit
   - [ ] Aplicar backpropagate() con parent pointers
   - **Tiempo inversión:** 2-3 horas
   - **Ganancia esperada:** 40-50% speedup TOTAL
   - **Prioridad:** MÁXIMA - hacer ANTES de profiling

2. **Poda estratégica de árbol MCTS**
   
   Tu especificación "ocultar nodos con >K simulaciones y <P% victorias":
   
   ```python
   def prune_unpromising_nodes(root, min_visits=5000, min_win_rate=0.30):
       """
       Eliminar nodos que tienen demasiadas simulaciones pero bajo win_rate
       Interpetación: No vale explorar más, es rama perdedora
       """
       
       def traverse(node):
           if node.visit_count > min_visits:
               win_rate = node.win_count / node.visit_count
               if win_rate < min_win_rate:
                   // Prune this subtree
                   node.children = {}
                   node.untried_moves = []
                   return "pruned"
           
           for child in node.children.values():
               traverse(child)
       
       traverse(root)
   
   Parámetros configurables:
   - min_visits: Threshold de "suficientemente explorado"
   - min_win_rate: Threshold de "demasiado malo"
   
   Tu propuesta: Hacer estos valores modificables
   Ejemplo: prune_unpromising_nodes(root, min_visits=1000, min_win_rate=0.25)
   ```

2. **Poda por profundidad**
   ```python
   def prune_by_depth(root, max_depth=10):
       """Limitar profundidad del árbol para controlar memoria"""
       
       def traverse(node, depth):
           if depth > max_depth:
               node.children = {}
               return
           
           for child in node.children.values():
               traverse(child, depth + 1)
       
       traverse(root, 0)
   ```

3. **Memory management**
   ```python
   Monitorear uso de memoria:
   - Logging de tamaño del árbol
   - Alert si > memoria_máx
   - Auto-prune si se acerca
   
   Estimación tamaño nodo (con __slots__):
   ~ 80-100 bytes por MCTSNode optimizado (era 200-300)
   Para árbol de 100k nodos: ~8-10 MB (antes ~20-30 MB)
   ```

4. **Profiling y validación**
   ```python
   import cProfile
   
   def profile_mcts_player():
       player = MCTSPlayer(config)
       
       pr = cProfile.Profile()
       pr.enable()
       
       for _ in range(100):
           board = HexBoard(5)
           player.play(board)
       
       pr.disable()
       pr.print_stats(sort='cumulative')
   
   Identificar post-optimizaciones:
   - Funciones todavía a optimizar
   - Bottlenecks residuales
   - Tradeoffs entre velocidad/memoria
   ```

5. **Paralelización (opcional - si tiempo permite)**
   ```
   Idea: Ejecutar múltiples simulaciones en paralelo
   
   Estrategia simple (Root Parallelization):
   - Múltiples threads ejecutan MCTS independientemente
   - Periódicamente sincronizar estadísticas en raíz
   
   Nota: Implementar SOLO si después de optimizaciones (1-5)
   todavía se tiene tiempo sobrante. No es crítico.
           
           for t in threads:
               t.join()
           
           return self._select_best_move(root)
       
       def _worker_mcts(self, root, num_sims):
           for _ in range(num_sims):
               self._mcts_iteration(root)
   ```
   
   **Nota:** Ver "MCTS Parallel algorithms" en literatura
   Orogunal por Chaslot et al (2008)

7. **Tuning final de parámetros**
   ```python
   Grid search de hyperparámetros:
   
   Para MCTS:
   - exploration_c: [0.05, 0.1, 0.15, 0.2]
   - rave_threshold: [500, 1000, 2000]
   - pw_constant: [0.5, 1.0, 1.5]
   
   Para MinMax:
   - max_depth: [4, 5, 6, 7]
   - time_limit: [1.0, 3.0, 5.0]
   
   Ejecutar torneo de configuraciones
   Seleccionar mejores
   ```

#### Criteria de éxito
- ✓ Poda implementada correctamente
- ✓ Uso de memoria monitoreado y controlado
- ✓ Profiling muestra mejoras
- ✓ Tiempo por movimiento consistentemente < 5 segundos

---

### FASE 9: Documentación Técnica en LaTeX
**Duración estimada:** 8-10 horas  
**Prioridad:** ALTA (deliverable final)

#### Objetivos
- Documentar completamente la implementación
- Justificar diseños y decisiones
- Proporcionar benchmarks y resultados
- Crear PDF profesional

#### Estructura del documento LaTeX

```latex
\documentclass{article}
\usepackage{amsmath, amssymb, graphicx, algorithm, algpseudocode}

\title{Jugador Virtual de Hex: 
        Implementación de MCTS y MinMax con Heurísticas}
\author{[Your Name]}
\date{2024}

\begin{document}

1. INTRODUCCIÓN
   - Descripción del juego Hex
   - Motivación: complejidad del juego
   - Objetivos del proyecto
   - Estructura del documento

2. FUNDAMENTOS TEÓRICOS
   2.1 Monte Carlo Tree Search (MCTS)
       - Algoritmo UCT (Upper Confidence bounds applied to Trees)
       - Fases: Selection, Expansion, Simulation, Backpropagation
       - Multi-armed Bandit problem
       - Referencias: Kocsis & Szepesvári (2006)
   
   2.2 Mejoras a MCTS
       2.2.1 RAVE (Rapid Action Value Estimation)
             - All-Moves-As-First statistics
             - Factor de decaimiento β(t)
             - Referencias: Gelly & Silver (2011)
       
       2.2.2 Estrategias Progresivas
             - Progressive Widening
             - Progressive Unpruning
             - Adaptación temporal de búsqueda
             - Referencias: Huang et al. (ICML)
   
   2.3 MinMax con Alpha-Beta Pruning
       - Algoritmo minimax
       - Pruning: Alpha-Beta cutoffs
       - Mejoras: Move ordering, Transposition tables
       - Iterative deepening
   
   2.4 Heurísticas para Hex
       2.4.1 Distance to Winning
       2.4.2 Connectivity Analysis
       2.4.3 Territory Control and Bridges
       2.4.4 Función de evaluación compuesta

3. IMPLEMENTACIÓN
   3.1 Arquitectura general
       - Diagrama de clases (UML)
       - Estructura de directorios
       - Módulos principales
   
   3.2 MCTS básico
       - Estructura MCTSNode
       - Selección UCT
       - Simulación aleatoria
       - Backpropagation
   
   3.3 Mejoras MCTS
       - Integration de RAVE
       - Progressive Widening
       - Estrategias de selección
   
   3.4 MinMax
       - Estructura de búsqueda
       - Transposition tables con Zobrist hashing
       - Iterative deepening
       - Integración de heurísticas
   
   3.5 Heurísticas
       - Implementación de cada heurística
       - Combinación y pesos
       - Generación de features

4. RESULTADOS EXPERIMENTALES
   4.1 Metodología
       - Setup experimental
       - Tamaños de tablero
       - Número de simulaciones
       - Tiempos permitidos
   
   4.2 Benchmarks MCTS
       - MCTS vs Random
       - MCTS + RAVE vs MCTS
       - Comparación estrategias progresivas
       - Tablas y gráficas
   
   4.3 Benchmarks MinMax
       - MinMax vs Random
       - MinMax vs MCTS (tableros pequeños)
       - Análisis de profundidad
       - Impacto de transposition tables
   
   4.4 Benchmarks heurísticas
       - Impacto individual de cada heurística
       - Combinación óptima
       - Convergencia vs playout puro
   
   4.5 Análisis de convergencia
       - Win_rate vs # simulaciones
       - Gráficas de aprendizaje
       - Eficiencia comparativa

5. ANÁLISIS Y DISCUSIÓN
   5.1 Características clave
       - Por qué MCTS es óptimo para grandes tableros
       - Por qué MinMax para tableros pequeños
       - Trade-offs discovery
   
   5.2 Limitaciones observadas
       - Factores que limitan performance
       - Problemas pendientes
       - Futuras mejoras
   
   5.3 Comparativa con literatura
       - Resultados vs papers citados
       - Innovaciones implementadas
       - Validación teórica

6. CONCLUSIONES
   - Resumen de logros
   - Contribuciones técnicas
   - Posibles extensiones

7. REFERENCIAS
   - Todos los papers citados
   - Libros de referencia
   - URLs consultadas

APÉNDICES:
A. Guía de instalación y uso
B. Parámetros de configuración
C. Logs de benchmarks completos
D. Código fuente destacado
```

#### Tareas específicas

1. **Tablas y gráficos**
   ```
   - Tabla 1: Win rates MCTS vs Random (tableros 3-7)
   - Tabla 2: Impacto de RAVE/Progressive strategies
   - Tabla 3: Comparación criterios de selección
   - Tabla 4: MinMax vs MCTS en diferentes tamaños
   - Figura 1: Convergencia de MCTS
   - Figura 2: Árbol MCTS ejemplo
   - Figura 3: Diagrama de fase MCTS
   - Figura 4: Performance vs tiempo
   ```

2. **Pseudocódigo formal**
   ```
   \begin{algorithm}
   \caption{MCTS with UCT}
   \begin{algorithmic}[1]
   ...
   \end{algorithmic}
   \end{algorithm}
   
   Para cada: MCTS, RAVE, MinMax, Heurísticos
   ```

3. **Ecuaciones matemáticas**
   ```latex
   UCT: Q(v)/N(v) + C√(ln(N_p)/N(v))
   
   RAVE: β(N) = √(τ/(3+N))
   
   Progressive: k(t) = C · t^p
   
   Evaluation: f(s) = Σ w_i · h_i(s)
   ```

4. **Figuras del proyecto**
   - Captura de pantidas del juego
   - Visualización de árboles
   - Gráficos de performance

5. **Compilación**
   ```bash
   pdflatex documento.tex
   bibtex documento
   pdflatex documento.tex  # dos pasadas para referencias
   ```

#### Criteria de éxito
- ✓ Documento de 25-35 páginas
- ✓ Compilación sin errores LaTeX
- ✓ Todas las figuras y tablas presentes
- ✓ Referencias completas
- ✓ PDF final legible y profesional

---

## Referencias Bibliográficas

### Papers primarios citados

**MCTS fundamentales:**
1. Kocsis, L., & Szepesvári, C. (2006). "Bandit based Monte-Carlo Tree Search". In European Conference on Machine Learning (ECML).
   - Define algoritmo UCT, base de MCTS moderno
   - Citado en: FASE 1 (UCT), FASE 2 (persistencia)

2. Gelly, S., & Silver, D. (2011). "Monte-Carlo Tree Search with Rapid Action Value Estimation". In IJCAI.
   - Introduce RAVE para convergencia más rápida
   - Citado en: FASE 3 (RAVE)

**Hex-specific:**
3. "Artificial Intelligence for the Hex Game.pdf" (Attachment)
   - Heurísticas específicas de Hex
   - Evaluacio de conectividad y territorio
   - Citado en: FASE 5, FASE 6, FASE 7

4. "Monte Carlo Tree Search in Hex.pdf" (Attachment)
   - Aplicación completa de MCTS a Hex
   - Estrategias de evaluación
   - Citado en: FASE 1, FASE 3, FASE 6

**Estrategias avanzadas:**
5. "PROGRESSIVE STRATEGIES FOR MONTE-CARLO TREE SEARCH.pdf" (Attachment)
   - Progressive Widening y Unpruning
   - Adaptación temporal de búsqueda
   - Citado en: FASE 4, FASE 8

6. "Monte Carlo Tree Search with Heuristic Evaluations.pdf" (Attachment)
   - Integración de heurísticas en playouts
   - Sesgo de política
   - Citado en: FASE 6

**MinMax clásico:**
7. Russell, S. J., & Norvig, P. (2020). "Artificial Intelligence: A Modern Approach" (4th ed.). Prentice Hall.
   - Alpha-Beta Pruning, Transposition Tables
   - Citado en: FASE 7, FASE 8

8. Knuth, D. W., & Moore, R. W. (1975). "An Analysis of Alpha-Beta Pruning", Artificial Intelligence.
   - Análisis teórico de pruning
   - Citado en: FASE 7

### Referencias adicionales

- Huang, A., et al. "Progressive Strategies for Monte-Carlo Tree Search"
- Chaslot, G., & Winands, M. (2008). "Parallel Monte-Carlo Tree Search"
- Cazenave, T. "Analysis of Monte-Carlo Tree Search", IJCAI 2009

---

## Consideraciones Técnicas

### Elecciones de diseño

1. **¿Por qué bifurcación MCTS/MinMax?**
   - Hex pequeño: Factor de rama bajo, MinMax exhaustivo es viable
   - Hex grande: Factor de rama exponencial, MCTS más eficiente
   - Punto de corte empírico: 7×7 (aproximadamente)

2. **¿Por qué UCT con C=0.1?**
   - Tu especificación default
   - Balanza reasonable entre exploration/exploitation para Hex
   - Típicamente se tunea entre [0.05, 0.2]

3. **¿Persistencia con seguridad primero?**
   - Evita bugs silenciosos de inconsistencia de estado
   - Cost: Reset ocasional (acceptable)
   - Benefit: Confiabilidad total

4. **¿Por qué RAVE antes de estrategias progresivas?**
   - Fundación más simple
   - Mejora measurable
   - Setup para investigación de sesgo

### Escalabilidad esperada

```
Performance anticipated:

Tablero 3×3:  500ms    (búsqueda casi exhaustiva)
Tablero 4×4:  1s       (MCTS bien convergido)
Tablero 5×5:  2-3s     (búsqueda profunda)
Tablero 6×6:  4-5s     (límite temporal)
Tablero 7×7:  5s+      (límite alcanzado)
Tablero 8×8:  timeout  (demasiado grande para 5 segundos)
```

### Dependencias

```
Python 3.8+
- threading (paralelización)
- numpy (opcional, para operaciones matriciales)
- matplotlib (gráficos de evaluación)
- numba (opcional, JIT compilation para hotspots)

Sin dependencias externas requeridas para funcionalidad core.
```

### Próximas mejoras (futuro)

1. GPU parallelization con CUDA
2. Deep Learning: Networks para evaluación
3. MCTS-supervised learning (AlphaGo-style)
4. Busca sin transposition tables (memoria limited)
5. Endgame solver exacto (tablebases)

---

## Conclusiones del Plan

Este plan detalla un desarrollo **incrementalista** que mantiene **siempre funcionalidad** minonal mientras agrega complejidad.

**Fases esenciales (core):**
- FASE 0: Infraestructura
- FASE 1: MCTS básico
- FASE 6: Heurísticas
- FASE 9: Documentación

**Fases de mejora opcional (depende de tiempo):**
- FASE 2: Persistencia
- FASE 3: RAVE
- FASE 4: Estrategias progresivas
- FASE 7: MinMax
- FASE 8: Optimización

**Estimación total:** 50-70 horas de desarrollo completo.

Puedes hacer checkpoints después de cada fase y ajustar scope según necesidad.

