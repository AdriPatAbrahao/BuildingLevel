"""
Defines configuration dataclasses and a factory class for creating
structural elements within a TQS model using the TQS API.
"""

from TQS import TQSModel, TQSUtil
from dataclasses import dataclass
from typing import List, Tuple, Optional
import traceback
from config.constants import (
    DEFAULT_SLAB_THICKNESS_CM,
    DEFAULT_SLAB_DEAD_LOAD_TF_M2,
    DEFAULT_SLAB_LIVE_LOAD_TF_M2,
    DEFAULT_SLAB_ANGLE_DEGREES,
    DEFAULT_SLAB_LOAD_CASE
)

# =============================================================================
# TQS Element Factory Class
# =============================================================================

class TQSElementFactory:
    """
    Provides static methods to create TQS structural elements (Columns, Beams, Slabs).
    This acts as a dedicated interface for element creation, abstracting the
    specific TQS API calls.
    """

    
    @staticmethod
    def create_polygonal_column(model: TQSModel, floor: TQSModel.Floor,
                                coords_list: List[Tuple[float, float]]) -> Optional[TQSModel.Column]:
        """
        Creates a polygonal column in the specified TQS floor.

        Args:
            model (TQSModel): The active TQS model instance.
            floor (TQSModel.Floor): The TQS floor object where the column will be inserted.
            coords_list (List[Tuple[float, float]]): List of (x, y) coordinate tuples
                defining the vertices of the column polygon in counter-clockwise or
                clockwise order (TQS handles orientation). Units are typically cm.
                Example: [(0, 0), (20, 0), (20, 50), (0, 50)]

        Returns:
            Optional[TQSModel.Column]: The created TQS Column object, or None if creation failed.
        """
        if not coords_list:
             TQSUtil.writef("Error creating column: Coordinate list is empty.")
             return None
        
        if len(coords_list) < 3: # A polygon needs at least 3 vertices
             TQSUtil.writef(f"Error (TQSElementFactory): Cannot create column, polygon needs at least 3 vertices, got {len(coords_list)}.")
             return None
        try:
            # Access global column data for default settings
            col_data = model.current.globalColumnData.columnData
            col_data.columnStarts = TQSModel.COLUMNSTART_NASCEDIRT # Default: Fixed support at start
            col_data.columnInsertion.insertionType = 1             # Default: Insert by corner
            col_data.columnInsertion.insertionCorner = 0           # Default: Use first corner (index 0)
           
            # Define geometry type
            col_geom = col_data.columnGeometry
            col_geom.sectionType = TQSModel.COLUMNTYPE_P           # P = Polygonal section

            # Define polygon vertices
            col_poly = col_data.columnPolygon
            col_poly.Clear() # Clear any previous polygon data
            for x_coord, y_coord in coords_list:
                col_poly.Enter(x_coord, y_coord)

            # Use the first coordinate for insertion point
            insert_x, insert_y = coords_list[0]

            # Create the column element on the specified floor
            new_column = floor.create.CreateColumn(insert_x, insert_y)
            if new_column:
                 pass
            else:
                 TQSUtil.writef(f"Warning: TQS floor.create.CreateColumn returned None at ({insert_x}, {insert_y}).")

            return new_column
            
        except Exception as e:
            TQSUtil.writef(f"Error creating polygonal column: {str(e)}")
            TQSUtil.writef(traceback.format_exc())
            return None

    @staticmethod
    def create_beam(floor: TQSModel.Floor, # model might not be needed if not accessing global data
                    start_node: Tuple[float, float], end_node: Tuple[float, float],
                    width_cm: float, height_cm: float,
                    perm_load_tf_m: float, live_load_tf_m: float) -> Optional[TQSModel.Beam]:
        """
        Creates a straight beam between two points in the specified TQS floor.

        Args:
            floor (TQSModel.Floor): The TQS floor object where the beam will be inserted.
            start_node (Tuple[float, float]): (x, y) coordinates of the beam's start point (cm).
            end_node (Tuple[float, float]): (x, y) coordinates of the beam's end point (cm).
            width_cm (float): Beam width (cm).
            height_cm (float): Beam height (cm).
            perm_load_tf_m (float): Permanent distributed load (tf/m - ton-force per meter).
            live_load_tf_m (float): Live distributed load (tf/m).

        Returns:
            Optional[TQSModel.Beam]: The created TQS Beam object, or None if creation failed.
        """
        if start_node == end_node:
            TQSUtil.writef(f"Warning: Skipping beam creation with zero length at {start_node}.")
            return None
        
        if width_cm <= 0 or height_cm <= 0:
            TQSUtil.writef(f"Error (TQSElementFactory): Beam dimensions must be positive. Got width={width_cm}, height={height_cm}.")
            return None

        try:
            # Set beam geometry defaults for the *next* beam to be created
            beam_data = floor.current.floorBeamData # Corrected to access floorBeamData
            beam_geom = beam_data.beamGeometry
            beam_geom.width = width_cm
            beam_geom.depth = height_cm
            
            # Set beam load defaults for the *next* beam to be created
            load_data = floor.current.floorLoadData 
            load = load_data.GetLoad(TQSModel.TPLOAD_CARVIG)
            load_case_index = DEFAULT_SLAB_LOAD_CASE
            load.SetMainLoad(load_case_index, perm_load_tf_m)
            load.SetLiveLoad(load_case_index, live_load_tf_m)
            
            # Define beam coordinates
            xy_coords = [start_node, end_node]

            # Create the beam element
            new_beam = floor.create.CreateBeam(xy_coords)

            if new_beam:
                 # TQSUtil.writef(f"Beam created successfully from {start_node} to {end_node}.")
                 pass # Keep console less verbose
            else:
                 TQSUtil.writef(f"Warning: TQS floor.create.CreateBeam returned None for {start_node} -> {end_node}.")

            return new_beam
            
        except Exception as e:
            TQSUtil.writef(f"Warning (TQSElementFactory): TQS floor.create.CreateBeam returned None for {start_node} -> {end_node}.")
            TQSUtil.writef(traceback.format_exc())
            return None

    @staticmethod
    def create_slab(floor: TQSModel.Floor,
                    insert_x_cm: float, insert_y_cm: float,
                    angle_degrees: float = DEFAULT_SLAB_ANGLE_DEGREES,
                    thickness_cm: float = DEFAULT_SLAB_THICKNESS_CM,
                    dead_load_tf_m2: float = DEFAULT_SLAB_DEAD_LOAD_TF_M2, # Renamed from perm_load
                    live_load_tf_m2: float = DEFAULT_SLAB_LIVE_LOAD_TF_M2
                   ) -> Optional[TQSModel.Slab]:
        """
        Creates a solid slab identified by a point inside it, using default parameters from SlabData.

        Args:
            floor (TQSModel.Floor): The TQS floor object where the slab will be inserted.
            insert_x_cm (float): X-coordinate of a point inside the desired slab area (cm).
            insert_y_cm (float): Y-coordinate of a point inside the desired slab area (cm).
            angle_degrees (float): Main reinforcement angle (degrees).
            thickness_cm (float): Slab thickness (cm).
            dead_load_tf_m2 (float): Dead (permanent) area load (tf/m²).
            live_load_tf_m2 (float): Live area load (tf/m²).

        Returns:
            Optional[TQSModel.Slab]: The created TQS Slab object, or None if creation failed.
        """
        if thickness_cm <= 0:
            TQSUtil.writef(f"Error (TQSElementFactory): Slab thickness must be positive. Got {thickness_cm}.")
            return None
        try:
            # Set slab geometry defaults for the *next* slab
            slab_data = floor.current.floorSlabData # Corrected to access floorSlabData
            slab_geom = slab_data.slabGeometry
            slab_geom.type = TQSModel.SLABTYPE_MACICA # Solid slab type
            slab_geom.thickness = thickness_cm

            # Set slab load defaults for the *next* slab
            load_data = floor.current.floorLoadData # Corrected to access floorLoadData
            load = load_data.GetLoad(TQSModel.TPLOAD_CARLAJ)
            load_case_index = DEFAULT_SLAB_LOAD_CASE
            load.SetMainLoad(load_case_index, dead_load_tf_m2) # Use dead_load from args
            load.SetLiveLoad(load_case_index, live_load_tf_m2) # Use live_load from args

            # Create the slab using the insertion point and angle
            # TQS automatically finds boundaries based on surrounding beams/walls.
            new_slab = floor.create.CreateSlab(insert_x_cm, insert_y_cm, angle_degrees)

            if new_slab:
                 # TQSUtil.writef(f"Slab created successfully at ({insert_x_cm}, {insert_y_cm}).") # Verbose
                 pass
            else:
                 TQSUtil.writef(f"Warning (TQSElementFactory): TQS floor.create.CreateSlab returned None for point ({insert_x_cm}, {insert_y_cm}). Check boundaries.")
            return new_slab

        except Exception as e:
            TQSUtil.writef(f"Error (TQSElementFactory) creating slab at ({insert_x_cm}, {insert_y_cm}): {str(e)}")
            TQSUtil.writef(traceback.format_exc())
            return None

  