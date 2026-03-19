"""Progressive Unpruning: Adaptive branching factor control."""

from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING

from shapely import node

if TYPE_CHECKING:
    from progressive_MCTS_player import _ProgressiveMCTSNode


class ExpandabilityManager:
    """
    Manages Progressive Unpruning: adaptive branching factor control per game phase.
    
    Progressive Unpruning restricts initial expansion and gradually liberalizes it
    as the search progresses. In opening phase with ~25 legal moves, unrestricted
    expansion creates a very wide tree. Progressive Unpruning favors depth in
    early stages and gradually broadens exploration as more information accumulates.
    
    Based on: Chaslot et al. (2008) "Progressive Strategies for Monte-Carlo Tree Search"
    - Section 3.2: Progressive Unpruning algorithm
    
    Formula (Chaslot 2008, page 9):
        k(n) = base + floor(C * sqrt(n))
    
    Where:
        - n = current node's visit_count
        - base = minimum expandable children for this phase
        - C = growth factor (phase-dependent)
        - sqrt(n) = smoothly increasing growth function
    
    Example (OPENING phase with base=1, C=0.5):
        n=0:    k(0) = 1
        n=100:  k(100) = 1 + floor(0.5 * 10) = 6
        n=400:  k(400) = 1 + floor(0.5 * 20) = 11
        n→∞:    k(∞) → ∞ (eventually expands all children)
    """
    
    # Expansion parameters: (base, C_factor) per game phase
    _EXPANSION_PARAMS = {
        "OPENING": {
            "base": 1,        # Highly restrictive initially
            "C_factor": 0.5,  # Slow growth
            "description": "Restrict deeply, grow slowly to favor depth over width",
        },
        "MIDGAME": {
            "base": 5,        # Moderate
            "C_factor": 1.0,  # Normal growth
            "description": "Moderate restriction for balanced width/depth exploration",
        },
        "ENDGAME": {
            "base": 999,      # Permissive: expand all children (tree naturally small)
            "C_factor": 0.0,  # Little to no growth: all children already considered
            "description": "No practical restriction (early endgame trees are small)",
        },
    }
    
    @staticmethod
    def get_max_expandable_children(phase: str, node_visits: int) -> int:
        """
        Calculate maximum number of children that can be expanded at a given node.
        
        Implements the formula from Chaslot 2008:
            k(n) = base + floor(C × sqrt(n))
        
        Args:
            phase: Game phase - "OPENING" | "MIDGAME" | "ENDGAME"
            node_visits: Current node's visit_count (number of times visited)
            
        Returns:
            Maximum number of children that should be expanded at this node visit count
            
        Examples:
            OPENING:  k(0)=1, k(4)=2, k(100)=6, k(400)=11
            MIDGAME:  k(0)=5, k(4)=7, k(100)=15, k(400)=25
            ENDGAME:  k(n)=999 for all n
        """
        if phase not in ExpandabilityManager._EXPANSION_PARAMS:
            phase = "MIDGAME"  # Default fallback
        
        params = ExpandabilityManager._EXPANSION_PARAMS[phase]
        base = params["base"]
        C = params["C_factor"]
        
        # Fórmula: k(n) = base + floor(C * sqrt(n))
        k_n = base + int(C * math.sqrt(node_visits))
        
        return k_n
    
    @staticmethod
    def is_expansion_allowed(phase: str, node: _ProgressiveMCTSNode) -> bool:
        """
        Determine whether a new child can be expanded at the given node.
        
        Logic:
        1. Calculate k(n) = maximum children allowed at this node
        2. Count currently expanded children
        3. Return True if count < k(n), False if expansion limit reached
        
        Rationale (Chaslot 2008, Section 3.2):
            "If the expansion limit is reached, the node selection strategy is 
            used instead. This creates a soft transition between expansion phase 
            and pure selection phase."
        
        Args:
            phase: Current game phase ("OPENING" | "MIDGAME" | "ENDGAME")
            node: _ProgressiveMCTSNode to evaluate
            
        Returns:
            Boolean - True if another child can be expanded, False if k(n) limit reached
        """
        # No limit at root level: explore all moves
        if node.parent is None:
            return True
    
        max_expandable = ExpandabilityManager.get_max_expandable_children(
            phase, 
            node.visit_count
        )
        current_children_count = len(node.children)
        
        return current_children_count < max_expandable
    
    @staticmethod
    def get_k_progression_table(phase: str) -> dict:
        """
        Generate a reference table showing k(n) values at key milestones.
        
        Useful for debugging, logging, and understanding Progressive Unpruning behavior.
        
        Args:
            phase: Game phase - "OPENING" | "MIDGAME" | "ENDGAME"
            
        Returns:
            Dictionary mapping visit counts to k(n) values:
            { n: k(n) for n in [0, 1, 4, 9, 25, 100, 225, 400] }
        """
        milestones = [0, 1, 4, 9, 25, 100, 225, 400]
        return {
            n: ExpandabilityManager.get_max_expandable_children(phase, n)
            for n in milestones
        }
