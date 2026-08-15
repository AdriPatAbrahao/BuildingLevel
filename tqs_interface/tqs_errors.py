import os
import ctypes
from dataclasses import dataclass
from typing import List, Tuple
from config.settings import BuildingConfig
from config.paths import TQS_OUTPUT_DIR


@dataclass
class ErrorData:
    elm_number: int
    error_header: str


class TQSErrorReader:
    """
    Read critical structural errors via TQS DLLs (NGERERRO / NMSGERRO).

    Attempts to load the DLLs using an environment-configured directory
    (`BUILDOPT_TQS_DLL_DIR`) with sensible fallbacks. Iterates over target
    folders within the TQS project and collects CLASSIFICATION==2 errors.
    """

    def __init__(self):
        self._ngererro = None
        self._nmsgerro = None
        self._dll_dir = None
        candidates = []
        env_dir = os.getenv("BUILDOPT_TQS_DLL_DIR", "")
        if env_dir:
            candidates.append(env_dir)
        base_dir = str(TQS_OUTPUT_DIR)
        candidates.append(base_dir)
        candidates.append(os.path.join(base_dir, "DLL"))
        candidates.append(os.path.join(base_dir, "bin"))
        candidates.append(r"T:\TQSW\EXEC\X64")
        for d in candidates:
            try:
                self._ngererro = ctypes.WinDLL(os.path.join(d, "NGERERRO.dll"))
                self._nmsgerro = ctypes.WinDLL(os.path.join(d, "NMSGERRO.dll"))
                self._dll_dir = d
                break
            except Exception:
                self._ngererro = None
                self._nmsgerro = None

    def _dlls_available(self) -> bool:
        return self._ngererro is not None and self._nmsgerro is not None

    def get_critical_errors(
        self, building_name: str = None, strict: bool = False
    ) -> List[ErrorData]:
        """
        Collect critical errors (classification==2) from three project folders.

        Parameters
        ----------
        building_name : str, optional
            Building slot to inspect.  Defaults to ``BuildingConfig.NAME``.
            Pass an explicit name when reading errors for a worker slot
            (e.g. ``"OptimBuilding_02"``).

        Returns
        -------
        List[ErrorData]
            Unique elements with critical errors across VIGAS, PILAR, ESPACIAL.
        """
        _name = building_name if building_name else BuildingConfig.NAME
        targets = [
            os.path.join(str(TQS_OUTPUT_DIR), _name, "Tipo", "VIGAS"),
            os.path.join(str(TQS_OUTPUT_DIR), _name, "PILAR"),
            os.path.join(str(TQS_OUTPUT_DIR), _name, "ESPACIAL"),
        ]

        collected: List[Tuple[int, str]] = []

        if not self._dlls_available():
            if strict:
                raise RuntimeError("TQS error-reader DLLs are unavailable.")
            return []

        # Define signatures (best-effort based on C# sample)
        ERR_OPEN = getattr(self._ngererro, "ERR_OPEN", None)
        ERR_CLOSE = getattr(self._ngererro, "ERR_CLOSE", None)
        ERR_NPROG = getattr(self._ngererro, "ERR_NPROG", None)
        ERR_POSPROG = getattr(self._ngererro, "ERR_POSPROG", None)
        ERR_LER = getattr(self._ngererro, "ERR_LER", None)
        ERR_HEAD = getattr(self._ngererro, "ERR_HEAD", None)
        ERR_LITPEL = getattr(self._ngererro, "ERR_LITPEL", None)

        ERRO_OPEN = getattr(self._nmsgerro, "ERRO_OPEN", None)
        ERRO_CLOSE = getattr(self._nmsgerro, "ERRO_CLOSE", None)
        ERRO_LER = getattr(self._nmsgerro, "ERRO_LER", None)
        ERRO_CLASS = getattr(self._nmsgerro, "ERRO_CLASS", None)
        ERRO_DESCR = getattr(self._nmsgerro, "ERRO_DESCR", None)

        if not all([ERR_OPEN, ERR_CLOSE, ERR_NPROG, ERR_POSPROG, ERR_LER, ERR_HEAD, ERR_LITPEL,
                    ERRO_OPEN, ERRO_CLOSE, ERRO_LER, ERRO_CLASS, ERRO_DESCR]):
            if strict:
                raise RuntimeError("TQS error-reader DLL API is incomplete.")
            return []

        # Basic argtypes/restype assumptions
        c_int_p = ctypes.POINTER(ctypes.c_int)
        ERR_OPEN.argtypes = [c_int_p]
        ERRO_OPEN.argtypes = [c_int_p]
        ERR_CLOSE.argtypes = []
        ERRO_CLOSE.argtypes = []
        ERR_NPROG.argtypes = [c_int_p]
        ERR_POSPROG.argtypes = [c_int_p, ctypes.c_char_p, ctypes.c_int, c_int_p]
        ERR_LER.argtypes = [c_int_p]
        ERR_HEAD.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, c_int_p, c_int_p]
        ERR_LITPEL.argtypes = [c_int_p]
        ERRO_LER.argtypes = [ctypes.c_char_p, ctypes.c_int, c_int_p]
        ERRO_CLASS.argtypes = [c_int_p]
        ERRO_DESCR.argtypes = [ctypes.c_char_p, ctypes.c_int]

        original_cwd = os.getcwd()
        visited_targets = 0
        for path in targets:
            if not os.path.isdir(path):
                continue
            visited_targets += 1
            try:
                os.chdir(path)
                istat = ctypes.c_int(0)
                ERRO_OPEN(ctypes.byref(istat))
                if istat.value != 0:
                    if strict:
                        raise RuntimeError(
                            f"ERRO_OPEN failed with status {istat.value} in '{path}'."
                        )
                    continue
                ERR_OPEN(ctypes.byref(istat))
                if istat.value != 0:
                    ERRO_CLOSE()
                    if strict:
                        raise RuntimeError(
                            f"ERR_OPEN failed with status {istat.value} in '{path}'."
                        )
                    continue

                numprogr = ctypes.c_int(0)
                ERR_NPROG(ctypes.byref(numprogr))
                for i in range(numprogr.value):
                    iprogr = ctypes.c_int(i + 1)
                    numerros = ctypes.c_int(0)
                    nomprog = ctypes.create_string_buffer(261)
                    ERR_POSPROG(ctypes.byref(iprogr), nomprog, 0, ctypes.byref(numerros))
                    for j in range(numerros.value):
                        ierro = ctypes.c_int(j + 1)
                        ERR_LER(ctypes.byref(ierro))
                        nomerro = ctypes.create_string_buffer(261)
                        nomsist = ctypes.create_string_buffer(261)
                        iele = ctypes.c_int(0)
                        itre = ctypes.c_int(0)
                        ERR_HEAD(nomerro, 0, nomsist, 0, ctypes.byref(iele), ctypes.byref(itre))
                        itpelm = ctypes.c_int(0)
                        ERR_LITPEL(ctypes.byref(itpelm))

                        ERRO_LER(nomerro, 0, ctypes.byref(istat))
                        iclass = ctypes.c_int(0)
                        ERRO_CLASS(ctypes.byref(iclass))
                        desc = ctypes.create_string_buffer(261)
                        ERRO_DESCR(desc, 0)
                        if iclass.value == 2:
                            error_header = desc.value.decode(errors="ignore").split("\x00")[0]
                            collected.append((iele.value, error_header))

            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"TQS error-reader failed in '{path}': {exc}"
                    ) from exc
            finally:
                try:
                    ERR_CLOSE()
                except Exception:
                    pass
                try:
                    ERRO_CLOSE()
                except Exception:
                    pass
                try:
                    os.chdir(original_cwd)
                except OSError:
                    pass

        if strict and visited_targets == 0:
            raise RuntimeError(
                f"No TQS error folders were found for building '{_name}'."
            )

        # Deduplicate
        unique = {}
        for elm, header in collected:
            unique[elm] = header
        return [ErrorData(elm_number=k, error_header=v) for k, v in unique.items()]
