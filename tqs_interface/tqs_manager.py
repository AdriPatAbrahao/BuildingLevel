from config.constants import DEFAULT_BEAM_DEAD_LOAD_TF_M, DEFAULT_BEAM_HEIGHT_CM, DEFAULT_BEAM_LIVE_LOAD_TF_M, DEFAULT_BEAM_WIDTH_CM
from config.settings import BuildingConfig
from typing import Optional, List, Dict, Tuple
from shapely.geometry import Polygon
from .tqs_model import TQSElementFactory
from .tqs_build import AddBuilding
import traceback
from TQS import TQSUtil, TQSModel


# Constants
NOMPLANTA_FUNDAC = "Fundacao"        # Foundation floor name
NOMPLANTA_TIPO = "Tipo"              # Typical floor name

class TQSModelManager:
    def __init__(self, building_name: str = BuildingConfig.NAME):
        """Initialize TQS model manager with building name"""
        self.building_name = building_name
        self.model: Optional[TQSModel.Model] = None
        
     
    def _initialize_tqs_model_instance(self) -> bool:
        """Initialize TQS model parameters"""
        try:
            self.model = TQSModel.Model()
            if self.model.file.OpenNewModel() != 0:
                TQSUtil.writef("Manager Error: OpenNewModel failed.")
                return False
            if self.model.file.Save() != 0: # Initial save to establish the model file
                TQSUtil.writef("Manager Error: Initial model save failed.")
                return False
            TQSUtil.writef("Manager: TQS model instance initialized and saved.")
            return True
        except Exception as e:
            TQSUtil.writef(f"Manager Error: Exception during TQS model instance initialization: {e}")
            TQSUtil.writef(traceback.format_exc())
            return False
        
            
    def _save_tqs_model(self) -> bool:
        """Saves the current TQS model. Renamed for clarity."""
        if not self.model:
            TQSUtil.writef("Manager Warning: No active TQS model to save.")
            return False # Or True, if not having a model isn't an error state here
        TQSUtil.writef("Manager: Saving TQS model...")
        try:
            if self.model.file.Save() != 0:
                TQSUtil.writef("Manager Error: Save TQS model failed.")
                return False
            TQSUtil.writef("Manager: TQS model saved successfully.")
            return True
        except Exception as e:
            TQSUtil.writef(f"Manager Error: Exception during TQS model save: {e}")
            TQSUtil.writef(traceback.format_exc())
            return False

    def _add_columns(self, column_polygons: List[Polygon]) -> bool:
        """
        Adds polygonal columns to the foundation floor of the current TQS model.

        Args:
            column_polygons (List[Polygon]): List of Shapely Polygon objects,
                                             where each polygon defines a column's cross-section.
                                             Coordinates are expected in cm.
        Returns:
            bool: True if all columns were added successfully, False otherwise.
        """
        if not self.model:
            TQSUtil.writef("Manager Error: Cannot add columns, TQS model not initialized.")
            return False
        if not column_polygons:
            TQSUtil.writef("Manager Info: No column polygons provided to add.")
            return True 
        
        TQSUtil.writef(f"Manager: Adding {len(column_polygons)} column groups to floor '{NOMPLANTA_FUNDAC}'...")

        try:
            # Get the foundation floor using correct name
            foundation_floor_name = self.model.floors.GetFloorName(1)
            floor = self.model.floors.GetFloor(foundation_floor_name if foundation_floor_name else NOMPLANTA_FUNDAC)
            
            if not floor:
                TQSUtil.writef(f"Manager Error: Could not get foundation floor '{NOMPLANTA_FUNDAC}'.")
                return False
            
            # Add columns
            for i, polygon in enumerate(column_polygons):
                # Convert Shapely polygon exterior coordinates to list of tuples
                coord_list = list(polygon.exterior.coords)
                # Factory method handles TQS API calls
                if not TQSElementFactory.create_polygonal_column(self.model, floor, coord_list):
                    TQSUtil.writef(f"Manager Error: Failed to create column {i+1} using factory.")
                    return False # Stop if one column fails
                
            TQSUtil.writef("Manager: Performing intersections on foundation floor...")
            floor.util.DoIntersections() # Perform intersections after adding all elements on the floor
            TQSUtil.writef("Manager: Columns added and intersections performed.")
            return True
        except Exception as e:
            TQSUtil.writef(f"Manager Error: Exception during column addition: {e}")
            TQSUtil.writef(traceback.format_exc())
            return False
        
    def _add_slabs(self, slab_insertion_points: List[Tuple[float, float]]) -> bool:
        """Adiciona as lajes ao pavimento tipo, APÓS as vigas terem sido processadas."""
        if not self.model: return False
        if not slab_insertion_points: return True
        TQSUtil.writef(f"Manager: Adicionando {len(slab_insertion_points)} lajes em '{NOMPLANTA_TIPO}'...")
        # Add Slabs
        try:
            floor = self.model.floors.GetFloor(NOMPLANTA_TIPO)
            if not floor:
                TQSUtil.writef(f"Manager Error: Could not get foundation floor '{NOMPLANTA_FUNDAC}'.")
                return False
            
            if slab_insertion_points: # Only proceed if there are slabs to add
                for i, (insert_x, insert_y) in enumerate(slab_insertion_points):
                    if not TQSElementFactory.create_slab(
                        floor, 
                        insert_x_cm=insert_x, insert_y_cm=insert_y
                    ):
                        TQSUtil.writef(f"Manager Error: Failed to create slab {i+1} at ({insert_x},{insert_y}) using factory.")
                        return False # Stop if one slab fails
                    
            TQSUtil.writef("Manager: Performing intersections on typical floor...")
            floor.util.DoIntersections()
            TQSUtil.writef("Manager: Slabs added and intersections performed.")
            return True
        
        except Exception as e:
            TQSUtil.writef(f"Manager Error: Exception during slab addition: {e}")
            TQSUtil.writef(traceback.format_exc())
            return False
        
    def _add_beams(self, beam_definitions: List[Dict]) -> bool: 
        """
        Adds beams to the typical floor ('Tipo') of the current TQS model.

        Args:
            beam_definitions (List[Dict]): List of beam definitions. Each dict should have
                                           'node_1': (x,y) and 'node_2': (x,y) in cm.
        Returns:
            bool: True if all beams were added successfully, False otherwise.
        """
        TQSUtil.writef(f"Manager: Adicionando {len(beam_definitions)} vigas em '{NOMPLANTA_TIPO}'...")
        try:
            floor = self.model.floors.GetFloor(NOMPLANTA_TIPO) 
            if not floor:
                TQSUtil.writef(f"Manager Error: Could not get typical floor '{NOMPLANTA_TIPO}'.")
                return False
            
            # Add Beams
            if beam_definitions: # Only proceed if there are beams to add
                for i, beam_def in enumerate(beam_definitions):
                    node1 = beam_def.get("node_1")
                    node2 = beam_def.get("node_2")

                    if not node1 or not node2:
                        TQSUtil.writef(f"Manager Warning: Skipping beam {i+1} due to missing node data: {beam_def}")
                        continue
                
                    if not TQSElementFactory.create_beam(
                        floor, # Pass model if create_beam needs it again
                        start_node=node1, end_node=node2,
                        width_cm=DEFAULT_BEAM_WIDTH_CM,
                        height_cm=DEFAULT_BEAM_HEIGHT_CM,
                        perm_load_tf_m=DEFAULT_BEAM_DEAD_LOAD_TF_M,
                        live_load_tf_m=DEFAULT_BEAM_LIVE_LOAD_TF_M
                    ):
                        TQSUtil.writef(f"Manager Error: Failed to create beam {i+1} using factory.")
                        return False # Stop if one beam fails
                floor.util.DoIntersections()
                return True            


        except Exception as e:
            TQSUtil.writef(f"Manager Error: Exception during beam addition: {e}")
            TQSUtil.writef(traceback.format_exc())
            return False


    def create_building_model_and_elements(self, column_polygons: List[Polygon],
                                         beam_definitions: List[Dict]) -> bool:
        """
        Orchestrates the complete creation of a new TQS building model,
        including setting up the building, initializing the model instance,
        adding structural elements, and saving.

        Args:
            column_polygons (List[Polygon]): List of Shapely Polygons for columns.
            beam_definitions (List[Dict]): List of dictionaries defining beams.

        Returns:
            bool: True if the entire process was successful, False otherwise.
        """
        TQSUtil.writef(f"\n--- TQSModelManager: Starting Full Building Model Creation for '{self.building_name}' ---")
        
        try:
            # --- 1. Close any existing model instance in this manager ---
            if self.model:
                TQSUtil.writef("Manager: Closing previous model instance...")
                self.model.file.Close() # Ensure it's properly closed
                self.model = None
                
            # --- 2. Create/Setup TQS Building Project ---
            TQSUtil.writef(f"Manager: Setting up TQS Building project '{self.building_name}'...")
            tqs_building_project = AddBuilding(self.building_name) # From tqs_build.py
            if not tqs_building_project:
                TQSUtil.writef(f"Manager Error: AddBuilding failed for '{self.building_name}'.")
                return False
            # Set the root folder for the TQS project
            # This step might be crucial for TQS to find/save files correctly.
            if tqs_building_project.RootFolder(self.building_name) != 0:
                TQSUtil.writef(f"Manager Error: Failed to set RootFolder for '{self.building_name}'.")
                # It's possible AddBuilding already does this or it's not always needed.
                # If it fails, it might indicate a path or permission issue for TQS.
                return False # Or decide if this is a critical failure
            TQSUtil.writef(f"Manager: TQS Building project '{self.building_name}' setup complete.")
                
            # --- 3. Initialize a new TQSModel.Model instance ---
            if not self._initialize_tqs_model_instance():
                # Error already logged by the helper
                return False
                
            slab_points = getattr(BuildingConfig, 'SLAB_COORDINATES', [])
            if not slab_points:
                TQSUtil.writef("Manager Warning: No slab insertion points defined in BuildingConfig.SLAB_COORDINATES. Slabs will not be added.")


            TQSUtil.writef("Manager: Adding structural elements...")
            if not self._add_columns(column_polygons):
                # Error logged by helper
                return False
            if not self._save_tqs_model():
                # Error logged by helper
                return False
            if not self._add_beams(beam_definitions):
                # Error logged by helper
                return False
            if not self._save_tqs_model():
                # Error logged by helper
                return False
            if not self._add_slabs(slab_points):
                # Error logged by helper
                return False
            TQSUtil.writef("Manager: Structural elements added.")
                
            # --- 5. Final Save of the Populated Model ---
            if not self._save_tqs_model():
                # Error logged by helper
                return False

            TQSUtil.writef(f"--- TQSModelManager: Full Building Model Creation for '{self.building_name}' Successful ---")
            return True
           
        except Exception as e:
            TQSUtil.writef(f"Error creating building model: {str(e)}")
            return False