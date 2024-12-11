from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

GPOPP = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 2), ('avg_pool_3x3', 4)], reduce_concat=[2, 3, 4, 5])