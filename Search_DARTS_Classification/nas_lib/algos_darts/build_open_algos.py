import sys
# from .gin_uncertainty_predictor_open_search import gin_uncertainty_predictor_search_open
# from .gin_predictor_open_search import gin_predictor_search_open
from .gp_predictor_open_search import gp_predictor_search_open
from .new_gp_predictor_open_search import new_gp_predictor_search_open
from .nas_bench_darts import nas_bench_darts_search_open


def build_open_algos(agent):
    return getattr(sys.modules[__name__], agent+'_search_open')