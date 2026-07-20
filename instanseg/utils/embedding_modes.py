"""embedding_modes.py
--------------------
Defines and validates the six supported embedding modes for InstanSeg.

Each mode string combines two orthogonal axes:

Geometric reference (what the embedding vector points toward)
    ``center``    Displacement from each pixel toward the instance centroid.
    ``border``    Displacement from each pixel toward its nearest instance border.
    ``combined``  Both center *and* border vectors are predicted simultaneously
                  (4-channel embedding for 2-D images).

Instance-separation strategy (how instances are differentiated at inference)
    ``seed``      Candidate anchors are extracted from a seed / border-distance
                  map and a pixel classifier is used per crop.
    ``cluster``   All foreground pixels are grouped directly via a boundary-aware
                  embedding-affinity graph — no explicit seeds required.

Accepted values
---------------
``center-seed``     Center vectors + seed-based differentiation (default / legacy).
``border-seed``     Border vectors + seed-based differentiation.
``center-cluster``  Center vectors + graph-clustering differentiation.
``border-cluster``  Border vectors + graph-clustering differentiation.
``combined-center`` Both vectors + seed-based differentiation.
``combined-cluster`` Both vectors + graph-clustering differentiation.

Any other value raises ``ValueError`` with a descriptive message.
"""

VALID_EMBEDDING_MODES = (
    "center-seed",
    "border-seed",
    "center-cluster",
    "border-cluster",
    "combined-center",
    "combined-cluster",
)

EMBEDDING_MODE_ERROR = (
    "embedding_mode must be one of: 'center-seed', 'border-seed', "
    "'center-cluster', 'border-cluster', 'combined-center', 'combined-cluster'"
)


def validate_embedding_mode(embedding_mode: str) -> str:
    """Raise ``ValueError`` if *embedding_mode* is not one of the six accepted values.

    Returns the mode string unchanged so callers can write::

        self.embedding_mode = validate_embedding_mode(embedding_mode)
    """
    if embedding_mode not in VALID_EMBEDDING_MODES:
        raise ValueError(EMBEDDING_MODE_ERROR)
    return embedding_mode


def embedding_uses_center(embedding_mode: str) -> bool:
    """Return ``True`` when the mode includes a center-relative embedding component.

    True for: ``center-seed``, ``center-cluster``, ``combined-center``,
    ``combined-cluster``.
    """
    validate_embedding_mode(embedding_mode)
    return embedding_mode in ("center-seed", "center-cluster", "combined-center", "combined-cluster")


def embedding_uses_border(embedding_mode: str) -> bool:
    """Return ``True`` when the mode includes a border-relative embedding component.

    True for: ``border-seed``, ``border-cluster``, ``combined-center``,
    ``combined-cluster``.
    """
    validate_embedding_mode(embedding_mode)
    return embedding_mode in ("border-seed", "border-cluster", "combined-center", "combined-cluster")


def embedding_uses_clustering(embedding_mode: str) -> bool:
    """Return ``True`` when the mode uses graph-clustering for instance separation.

    True for: ``center-cluster``, ``border-cluster``, ``combined-cluster``.
    For these modes the seed map is *not* used as the primary instance-separation
    mechanism; instead ``cluster_embedding_pixels`` groups pixels directly.
    """
    validate_embedding_mode(embedding_mode)
    return embedding_mode in ("center-cluster", "border-cluster", "combined-cluster")


def embedding_vector_channels(embedding_mode: str, dim_coords: int = 2) -> int:
    """Return the number of embedding vector channels for *embedding_mode*.

    Combined modes predict two vector fields (center *and* border), so they
    require ``2 * dim_coords`` channels.  All other modes use ``dim_coords``
    channels (one vector field).

    Parameters
    ----------
    embedding_mode:
        One of the six accepted mode strings.
    dim_coords:
        Spatial dimensionality of the coordinate system (2 for 2-D images).
    """
    validate_embedding_mode(embedding_mode)
    if embedding_mode in ("combined-center", "combined-cluster"):
        return dim_coords * 2
    return dim_coords
