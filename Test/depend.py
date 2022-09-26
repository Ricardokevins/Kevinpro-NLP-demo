import numpy
import numpy as np
#生成正态分布
x = numpy.round(numpy.random.normal(10, 0.2, size=(10240,1)),5).reshape(-1)
x2 = numpy.round(numpy.random.normal(15, 0.2, size=(10240,1)),5).reshape(-1)
x3 = numpy.round(numpy.random.normal(15, 0.5, size=(10240,1)),5).reshape(-1)
x4 = numpy.round(numpy.random.normal(10, 0.2, size=(10240,1)),5).reshape(-1)




from scipy.stats import pearsonr

pccs,_ = pearsonr(x.tolist(), x2.tolist())
print(pccs)

pccs,_ = pearsonr(x.tolist(), x3.tolist())
print(pccs)

pccs,_ = pearsonr(x.tolist(), x4.tolist())
print(pccs)