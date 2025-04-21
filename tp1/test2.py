import template_funciones as tf
import numpy as np
import scipy

#test LU
test1=[[2,1,2,3],[4,3,3,4],[-2,2,-4,-12],[4,1,8,-3]]
L,U=tf.calculaLU(np.asarray(test1))
print(L , U)

#test inversa
