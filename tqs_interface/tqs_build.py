# coding: latin-1
#
#    ModeloEdificio.py
#    Script para modelagem de edifício com lajes, vigas e pilares no TQS
#-----------------------------------------------------------------------------
from TQS import TQSBuild, TQSUtil

def Delim():
    TQSUtil.writef('-' * 79)
#-----------------------------------------------------------------------------

def AddBuilding(nome_edificio):
    """
    Cria um novo edifício no TQS
    """
    try:
        building = TQSBuild.Building()
        istat = building.file.Open(nome_edificio)
        if istat != 0:
            TQSUtil.writef(f"Não foi possível criar o edifício [{nome_edificio}]")
            return None
        return building
    except Exception as e:
        TQSUtil.ShowException(e)
        return None