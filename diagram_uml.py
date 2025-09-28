# To generate this diagram, the 'graphviz' library is required: pip install graphviz
from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Vertical UML Class Diagram of the Optimization Framework')

# --- GLOBAL ATTRIBUTES FOR VERTICAL (A4) LAYOUT ---
# 'TB' (Top to Bottom) is the key change for vertical orientation.
# 'ortho' splines create clean, 90-degree lines suitable for architectural diagrams.
dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.8')
dot.attr('node', shape='record', style='rounded', fontname='Helvetica')
dot.attr('edge', fontname='Helvetica', fontsize='10')

# --- EXECUTION PHASE LAYER (TOP LEVEL) ---
with dot.subgraph(name='cluster_phases') as c:
    c.attr(label='Execution Phase Layer', style='rounded', color='black', rank='same')
    c.node('Phase1', '{<title>Phase 1: Training |<main>main.py}', tooltip='Orchestrates data generation and model training.')
    c.node('Phase2', '{<title>Phase 2: Optimization |<main>run_optimization.py}', tooltip='Executes the optimization using the trained model.')

# --- BUSINESS LOGIC LAYER ---
with dot.subgraph(name='cluster_logic') as c:
    c.attr(label='Business Logic Layer', style='rounded', color='grey')
    c.node('BuildingOptimizer', '{BuildingOptimizer|+ run_optimization()}', tooltip='Main class for Phase 1.')
    c.node('GeneticOptimizer', '{GeneticOptimizer|+ run()}', tooltip='Main class for Phase 2.')
    c.node('ObjectiveFunction', '{ObjectiveFunction|+ calculate_cost()}', tooltip='Calculates the cost of a design solution.')
    c.node('BuildingInference', '{BuildingInference|+ predict_from_csv()}', tooltip='Performs predictions with the loaded surrogate model.')

# --- CORE COMPONENTS & DATA ABSTRACTION LAYER ---
with dot.subgraph(name='cluster_core') as c:
    c.attr(label='Core Components & Data Abstraction Layer', style='rounded', color='lightblue')
    c.node('TQSModelManager', '{TQSModelManager|+ create_building_model()}', tooltip='Interfaces with the TQS API.')
    c.node('LengthProcessor', '{LengthProcessor|+ process_segments()}', tooltip='Processes the parametric vector input.')
    c.node('FeatureEngineer', '{FeatureEngineer|+ extract_features()}', tooltip='Extracts the feature vector from geometry.')
    c.node('NeuralNetworkManager', '{NeuralNetworkManager|+ train()\n+ predict()}', tooltip='Manages the ML model lifecycle.')
    c.node('ExperimentManager', '{ExperimentManager|+ log_metadata()}', tooltip='Manages the storage of experiment artifacts.')

# --- RELATIONSHIPS BETWEEN LAYERS ---

# Connections from Phase Layer to Business Logic Layer
dot.edge('Phase1:main', 'BuildingOptimizer', label='orchestrates')
dot.edge('Phase2:main', 'GeneticOptimizer', label='orchestrates')

# Connections within the Business Logic Layer
dot.edge('BuildingOptimizer', 'BuildingInference', label='uses for\nvalidation', style='dashed', dir='back')
dot.edge('GeneticOptimizer', 'ObjectiveFunction', label='minimizes')
dot.edge('ObjectiveFunction', 'BuildingInference', label='uses for\nprediction')

# Connections from Business Logic Layer to Core Components Layer
dot.edge('BuildingOptimizer', 'TQSModelManager')
dot.edge('BuildingOptimizer', 'LengthProcessor')
dot.edge('BuildingOptimizer', 'NeuralNetworkManager')
dot.edge('BuildingOptimizer', 'ExperimentManager')

dot.edge('BuildingInference', 'NeuralNetworkManager')
dot.edge('BuildingInference', 'FeatureEngineer')
dot.edge('FeatureEngineer', 'LengthProcessor')


# Render the graph (saves as PDF or PNG to be embedded in the thesis)
# To execute, uncomment the line below in a Python environment:
# dot.render('uml_diagram_vertical_english', format='pdf', view=True)
print("Python code for the vertical UML diagram has been generated. Execute it in an environment with the 'graphviz' library to produce the image file.")
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


dot.render('uml_diagram', format='png', view=True)
print("Código para diagrama UML gerado. Execute-o em um ambiente com a biblioteca 'graphviz' para visualizar a imagem.")

